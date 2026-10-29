// Layout of the duck's actor MLP (61 -> 512 -> 256 -> 128 -> 14) onto the brain shape.
import { regionAt, type RegionId } from "./anatomy";

export type Rec = {
  t: number[];
  cmd: number[][];
  L: number[][][]; // [layer][frame][neuron] values * scale
  edges: [number, number, number][][][]; // [layerPair][frame][k] = [src, dst, sign]
  base: number[][]; // x y z qw qx qy qz vx_body wz_body
  meta: {
    ckpt: string; fps: number; dt: number; scale: number;
    obs_terms: [string, number][]; act_joints: string[]; layer_sizes: number[];
    plan: number[][]; topk_edges_per_layer: number;
  };
};

export const JOINT_KO: Record<string, string> = {
  left_hip_yaw: "왼쪽 엉덩이 비틀기",
  left_hip_roll: "왼쪽 엉덩이 옆벌림",
  left_hip_pitch: "왼쪽 엉덩이 앞뒤",
  left_knee: "왼쪽 무릎",
  left_ankle: "왼쪽 발목",
  neck_pitch: "목 앞뒤",
  head_pitch: "머리 끄덕",
  head_yaw: "머리 좌우",
  head_roll: "머리 갸웃",
  right_hip_yaw: "오른쪽 엉덩이 비틀기",
  right_hip_roll: "오른쪽 엉덩이 옆벌림",
  right_hip_pitch: "오른쪽 엉덩이 앞뒤",
  right_knee: "오른쪽 무릎",
  right_ankle: "오른쪽 발목",
};

const OBS_KO: Record<string, string> = {
  base_ang_vel: "몸 회전 속도",
  projected_gravity: "중력 방향(기울기)",
  joint_pos: "관절 각도",
  joint_vel: "관절 속도",
  actions: "직전 출력",
  command: "걷기 명령",
  head_command: "머리 명령",
  body_command: "몸 자세 명령",
};
const AXES = ["x", "y", "z", "w"];

export type LayerInfo = { id: number; region: RegionId | "mixed"; name: string; sub: string; color: string };

export const LAYERS: LayerInfo[] = [
  { id: 0, region: "mixed", name: "입력층 61", sub: "감각: 시상·두정엽·후두엽", color: "#e0a84a" },
  { id: 1, region: "frontal", name: "은닉층 1 · 512", sub: "전두엽", color: "#d24bd8" },
  { id: 2, region: "temporal", name: "은닉층 2 · 256", sub: "측두엽", color: "#2fbf7a" },
  { id: 3, region: "cerebellum", name: "은닉층 3 · 128", sub: "소뇌", color: "#c45a32" },
  { id: 4, region: "stem", name: "출력층 14", sub: "뇌간 → 관절", color: "#6a7cff" },
];

export type NeuronPt = { layer: number; idx: number; region: RegionId; x: number; y: number; z: number; label: string };

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const BOX: Record<RegionId, [number, number, number, number, number, number]> = {
  // cx cy cz sx sy sz sampling box
  frontal: [0.6, 0.05, 0, 1.3, 1.8, 1.5],
  parietal: [-0.1, 0.5, 0, 1.4, 0.9, 1.5],
  occipital: [-0.8, 0.1, 0, 0.9, 1.4, 1.4],
  temporal: [0.4, -0.25, 0, 1.4, 0.9, 1.5],
  thalamus: [0, 0.02, 0, 0.44, 0.4, 0.44],
  cerebellum: [-0.78, -0.58, 0, 1.0, 0.8, 1.04],
  stem: [-0.28, -0.8, 0, 0.5, 1.4, 0.4],
};

function sampleRegion(rng: () => number, region: RegionId, n: number, minD: number) {
  const b = BOX[region];
  const out: [number, number, number][] = [];
  let d = minD;
  let guard = 0;
  let fails = 0;
  while (out.length < n && guard < n * 800) {
    guard++;
    if (fails > 150) { d *= 0.9; fails = 0; }
    const p: [number, number, number] = [b[0] + (rng() - 0.5) * b[3], b[1] + (rng() - 0.5) * b[4], b[2] + (rng() - 0.5) * b[5]];
    if (regionAt(p[0], p[1], p[2]) !== region) continue;
    let ok = true;
    for (const q of out) {
      if ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2 < d * d) { ok = false; break; }
    }
    if (ok) { out.push(p); fails = 0; } else fails++;
  }
  return out;
}

function stemX(y: number) {
  const t = Math.max(0, Math.min(1, (y + 0.2) / -1.25));
  return -0.22 - t * 0.12;
}

export function buildLayout(rec: Rec): NeuronPt[] {
  const rng = mulberry32(2026);
  const pts: NeuronPt[] = [];
  const joints = rec.meta.act_joints;
  // ---- input layer: group by observation meaning
  const inputRegion = (term: string): RegionId =>
    term === "joint_pos" || term === "joint_vel" ? "parietal" : term === "actions" ? "occipital" : "thalamus";
  const groups = new Map<RegionId, { term: string; k: number; idx: number }[]>();
  let idx = 0;
  for (const [term, n] of rec.meta.obs_terms) {
    for (let k = 0; k < n; k++) {
      const r = inputRegion(term);
      const list = groups.get(r) ?? [];
      list.push({ term, k, idx });
      groups.set(r, list);
      idx++;
    }
  }
  for (const [r, list] of groups) {
    const ps = sampleRegion(rng, r, list.length, r === "thalamus" ? 0.06 : 0.12);
    list.forEach((it, i) => {
      const p = ps[i] ?? ps[i % ps.length];
      const perJoint = it.term === "joint_pos" || it.term === "joint_vel" || it.term === "actions";
      const sub = perJoint ? JOINT_KO[joints[it.k]] ?? joints[it.k] : AXES[it.k] ?? String(it.k);
      pts.push({ layer: 0, idx: it.idx, region: r, x: p[0], y: p[1], z: p[2], label: `입력 · ${OBS_KO[it.term] ?? it.term} · ${sub}` });
    });
  }
  // ---- hidden layers
  const hidden: [number, RegionId, number][] = [[1, "frontal", 0.05], [2, "temporal", 0.05], [3, "cerebellum", 0.05]];
  for (const [layer, region, d] of hidden) {
    const n = rec.meta.layer_sizes[layer];
    const ps = sampleRegion(rng, region, n, d);
    for (let i = 0; i < n; i++) {
      const p = ps[i] ?? ps[i % ps.length];
      pts.push({ layer, idx: i, region, x: p[0], y: p[1], z: p[2], label: `${LAYERS[layer].name.split(" ·")[0]} · 뉴런 #${i}` });
    }
  }
  // ---- output layer: one neuron per actuated joint, laid along the brain stem
  joints.forEach((j, i) => {
    let y: number, z: number, xo = 0;
    if (j.startsWith("left_") || j.startsWith("right_")) {
      const order = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"];
      const k = order.findIndex((o) => j.endsWith(o));
      y = -0.5 - k * 0.19;
      z = j.startsWith("left_") ? 0.075 : -0.075;
    } else {
      const order = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"];
      const k = order.indexOf(j);
      y = -0.24 - Math.floor(k / 2) * 0.12;
      z = k % 2 === 0 ? 0.06 : -0.06;
      xo = 0.02;
    }
    pts.push({ layer: 4, idx: i, region: "stem", x: stemX(y) + xo, y, z, label: `출력 · ${JOINT_KO[j] ?? j} 목표 각도` });
  });
  return pts;
}
