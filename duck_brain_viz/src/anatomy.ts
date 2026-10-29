export type RegionId =
  | "frontal"
  | "parietal"
  | "occipital"
  | "temporal"
  | "thalamus"
  | "cerebellum"
  | "stem";

export type RegionInfo = { id: RegionId; label: string; color: string };

export const REGIONS: RegionInfo[] = [
  { id: "frontal", label: "전두엽", color: "#d24bd8" },
  { id: "parietal", label: "두정엽", color: "#7a4ae8" },
  { id: "occipital", label: "후두엽", color: "#3d7ef0" },
  { id: "temporal", label: "측두엽", color: "#2fbf7a" },
  { id: "thalamus", label: "시상", color: "#e0a84a" },
  { id: "cerebellum", label: "소뇌", color: "#c45a32" },
  { id: "stem", label: "뇌간", color: "#6a7cff" },
];

export type Tract = {
  positions: Float32Array;
  colors: Float32Array;
  seeds: Float32Array;
};

export type Neuron = {
  region: RegionId;
  x: number;
  y: number;
  z: number;
};

export type Axon = {
  from: number;
  to: number;
  pts: Float32Array;
  length: number;
};

const RUST: RGB = [0.77, 0.32, 0.16];
const GOLD: RGB = [0.86, 0.62, 0.26];
const GREEN: RGB = [0.18, 0.75, 0.42];
const MAGENTA: RGB = [0.82, 0.28, 0.74];
const VIOLET: RGB = [0.48, 0.26, 0.9];
const BLUE: RGB = [0.28, 0.45, 0.95];
const PEACH: RGB = [0.92, 0.62, 0.48];

type RGB = [number, number, number];
type V3 = [number, number, number];

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(v: number, a: number, b: number) {
  return Math.max(a, Math.min(b, v));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function mix3(a: RGB, b: RGB, t: number): RGB {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

function add3(a: RGB, b: RGB, w: number): RGB {
  return [a[0] + b[0] * w, a[1] + b[1] * w, a[2] + b[2] * w];
}

function len3(x: number, y: number, z: number) {
  return Math.hypot(x, y, z);
}

function norm(v: V3): V3 {
  const l = len3(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function srgbToLinear(c: number) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function lin(c: RGB): RGB {
  return [srgbToLinear(c[0]), srgbToLinear(c[1]), srgbToLinear(c[2])];
}

function hash3(ix: number, iy: number, iz: number) {
  let n = Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + Math.imul(iz, 1274126177);
  n = (n ^ (n >>> 13)) >>> 0;
  n = Math.imul(n, 1274126177);
  return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
}

function noise3(x: number, y: number, z: number) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const xf = x - x0;
  const yf = y - y0;
  const zf = z - z0;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const w = zf * zf * (3 - 2 * zf);
  const n000 = hash3(x0, y0, z0);
  const n100 = hash3(x0 + 1, y0, z0);
  const n010 = hash3(x0, y0 + 1, z0);
  const n110 = hash3(x0 + 1, y0 + 1, z0);
  const n001 = hash3(x0, y0, z0 + 1);
  const n101 = hash3(x0 + 1, y0, z0 + 1);
  const n011 = hash3(x0, y0 + 1, z0 + 1);
  const n111 = hash3(x0 + 1, y0 + 1, z0 + 1);
  const x00 = lerp(n000, n100, u);
  const x10 = lerp(n010, n110, u);
  const x01 = lerp(n001, n101, u);
  const x11 = lerp(n011, n111, u);
  return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

function curl(x: number, y: number, z: number): V3 {
  const e = 0.18;
  const n1 = noise3(x, y + e, z) - noise3(x, y - e, z);
  const n2 = noise3(x, y, z + e) - noise3(x, y, z - e);
  const n3 = noise3(x + e, y, z) - noise3(x - e, y, z);
  const n4 = noise3(x, y + e, z + e) - noise3(x, y - e, z - e);
  return [(n1 - n2) * 2.2, (n2 - n3) * 2.2, (n3 - n4) * 1.4];
}

function cerebrumR(x: number, y: number, z: number) {
  let nx = (x - 0.02) / 1.18;
  let ny = (y - 0.06) / 0.86;
  const nz = z / 0.7;
  if (x < -0.35) nx *= 0.9;
  const temporal = Math.exp(-((x - 0.15) ** 2) / 0.55) * Math.exp(-((y + 0.05) ** 2) / 0.12);
  ny -= temporal * 0.22;
  if (y < -0.15 && x > 0.15) ny *= 0.86;
  return nx * nx + ny * ny + nz * nz;
}

function inCerebrum(x: number, y: number, z: number) {
  return cerebrumR(x, y, z) < 1;
}

function inCerebellum(x: number, y: number, z: number) {
  const nx = (x + 0.78) / 0.5;
  const ny = (y + 0.58) / 0.4;
  const nz = z / 0.52;
  const fold = 0.06 * Math.sin(y * 28) * Math.sin(x * 18);
  return nx * nx + ny * ny + nz * nz < 1 + fold;
}

function stemCenter(y: number) {
  const t = clamp((y + 0.2) / -1.25, 0, 1);
  return -0.22 - t * 0.12;
}

function inStem(x: number, y: number, z: number) {
  if (y > -0.12 || y < -1.48) return false;
  const cx = stemCenter(y);
  const r = 0.2 - clamp((-0.12 - y) / 1.3, 0, 1) * 0.06;
  return (x - cx) ** 2 + z ** 2 < r * r;
}

export function insideBrain(x: number, y: number, z: number) {
  return inCerebrum(x, y, z) || inCerebellum(x, y, z) || inStem(x, y, z);
}

function ccPoint(t: number): V3 {
  const x = -0.58 + t * 1.28;
  const y = 0.02 + Math.sin(t * Math.PI) * 0.4;
  return [x, y, 0];
}

function cingPoint(t: number): V3 {
  const x = -0.72 + t * 1.55;
  const y = 0.16 + Math.sin(t * Math.PI) * 0.5;
  return [x, y, 0];
}

function infPoint(t: number): V3 {
  const x = -0.35 + t * 1.05;
  const y = -0.18 + Math.sin(t * Math.PI) * 0.12;
  return [x, y, 0];
}

function nearestT(x: number, y: number, curve: (t: number) => V3) {
  let best = 0;
  let bestD = 1e9;
  for (let i = 0; i <= 28; i++) {
    const t = i / 28;
    const p = curve(t);
    const d = (p[0] - x) ** 2 + (p[1] - y) ** 2 + p[2] ** 2 * 0.2;
    if (d < bestD) {
      bestD = d;
      best = t;
    }
  }
  return { t: best, d: Math.sqrt(bestD) };
}

function tangentAt(t: number, curve: (t: number) => V3): V3 {
  const a = curve(clamp(t - 0.02, 0, 1));
  const b = curve(clamp(t + 0.02, 0, 1));
  return norm([b[0] - a[0], b[1] - a[1], 0.15 * Math.sin(t * 9)]);
}

function cerebrumNormal(x: number, y: number, z: number): V3 {
  const e = 0.04;
  const dx = cerebrumR(x + e, y, z) - cerebrumR(x - e, y, z);
  const dy = cerebrumR(x, y + e, z) - cerebrumR(x, y - e, z);
  const dz = cerebrumR(x, y, z + e) - cerebrumR(x, y, z - e);
  return norm([dx, dy, dz]);
}

export function flowAt(x: number, y: number, z: number): V3 {
  let vx = 0;
  let vy = 0;
  let vz = 0;
  const cc = nearestT(x, y, ccPoint);
  const cing = nearestT(x, y, cingPoint);
  const inf = nearestT(x, y, infPoint);
  const wCc = Math.exp(-(cc.d * cc.d) / 0.02);
  const wCing = Math.exp(-(cing.d * cing.d) / 0.035);
  const wInf = Math.exp(-(inf.d * inf.d) / 0.03);
  const tCc = tangentAt(cc.t, ccPoint);
  const tCing = tangentAt(cing.t, cingPoint);
  const tInf = tangentAt(inf.t, infPoint);
  vx += tCc[0] * wCc * 1.6;
  vy += tCc[1] * wCc * 1.6;
  vz += tCc[2] * wCc * 0.3;
  vx += tCing[0] * wCing * 1.25;
  vy += tCing[1] * wCing * 1.25;
  vx += tInf[0] * wInf;
  vy += tInf[1] * wInf;

  if (inStem(x, y, z) || (y < -0.05 && Math.abs(x + 0.2) < 0.28 && Math.abs(z) < 0.22)) {
    const ws = y < -0.12 ? 1.4 : 0.45;
    vy -= ws;
    vx += -0.05 * ws;
  }

  if (inCerebellum(x, y, z)) {
    vx += Math.sin(y * 26) * 1.1;
    vy += 0.15;
    vz += Math.cos(x * 22) * 0.35;
  }

  const r = cerebrumR(x, y, z);
  if (r > 0.72 && r < 1.15 && inCerebrum(x, y, z)) {
    const n = cerebrumNormal(x, y, z);
    const shell = (r - 0.72) / 0.43;
    const fold = Math.sin(x * 11 + y * 4) * n[1] + Math.cos(y * 9) * n[0];
    vx += (n[0] * 0.85 + fold * 0.35) * shell;
    vy += (n[1] * 0.85 + Math.sin(z * 8 + x) * 0.2) * shell;
    vz += n[2] * 0.45 * shell;
  }

  const c = curl(x * 1.7, y * 1.7, z * 1.7);
  vx += c[0] * 0.28;
  vy += c[1] * 0.28;
  vz += c[2] * 0.22;
  return norm([vx, vy, vz]);
}

function paint(dir: V3, x: number, y: number, z: number, rand: number): RGB {
  const cc = nearestT(x, y, ccPoint);
  const cing = nearestT(x, y, cingPoint);
  let col: RGB = [0.15, 0.15, 0.18];
  let w = 0.001;
  const push = (src: RGB, weight: number) => {
    col = add3(col, src, weight);
    w += weight;
  };
  push(RUST, Math.exp(-(cc.d * cc.d) / 0.018) * 1.8);
  push(mix3(GREEN, GOLD, rand), Math.exp(-(cing.d * cing.d) / 0.03) * 1.3);
  if (inCerebellum(x, y, z)) push(mix3(VIOLET, RUST, rand), 1.1);
  if (inStem(x, y, z) || y < -0.35) push(mix3(BLUE, VIOLET, 0.35), 1.15);
  const shell = cerebrumR(x, y, z);
  if (shell > 0.78) {
    const cortical = mix3(MAGENTA, VIOLET, rand);
    const tipped = y > 0.35 || rand > 0.72 ? mix3(cortical, GOLD, 0.45) : cortical;
    push(mix3(tipped, PEACH, dir[1] > 0.6 ? 0.25 : 0), 0.9 + (shell - 0.78));
  }
  const ap = Math.abs(dir[0]);
  const si = Math.abs(dir[1]);
  const lr = Math.abs(dir[2]);
  push(GREEN, ap * 0.35);
  push(BLUE, si * 0.28);
  push(MAGENTA, lr * 0.4);
  return lin([clamp(col[0] / w, 0, 1), clamp(col[1] / w, 0, 1), clamp(col[2] / w, 0, 1)]);
}

function pushSeg(
  pos: number[],
  col: number[],
  seed: number[],
  a: V3,
  b: V3,
  c: RGB,
  s: number,
) {
  pos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  col.push(c[0], c[1], c[2], c[0], c[1], c[2]);
  seed.push(s, s);
}

function traceFiber(rng: () => number, origin: V3, steps: number, step: number, allowOut: number) {
  const pts: V3[] = [origin];
  let p = origin;
  let outside = 0;
  const flip = rng() < 0.5 ? 1 : -1;
  for (let i = 0; i < steps; i++) {
    const dir = flowAt(p[0], p[1], p[2]);
    const wob = curl(p[0] * 3 + i, p[1] * 3, p[2] * 3);
    const nx = p[0] + (dir[0] * flip + wob[0] * 0.12) * step;
    const ny = p[1] + (dir[1] * flip + wob[1] * 0.12) * step;
    const nz = p[2] + (dir[2] * flip + wob[2] * 0.12) * step;
    if (!insideBrain(nx, ny, nz)) {
      outside += 1;
      if (outside > allowOut) break;
    } else {
      outside = 0;
    }
    p = [nx, ny, nz];
    pts.push(p);
  }
  return pts;
}

function sampleOrigin(rng: () => number): { p: V3; kind: "shell" | "cc" | "cere" | "stem" | "deep" } {
  const roll = rng();
  if (roll < 0.58) {
    const theta = rng() * Math.PI * 2;
    const phi = Math.acos(2 * rng() - 1);
    const fold = 1 + 0.045 * Math.sin(theta * 7) * Math.sin(phi * 5);
    const jitter = (0.9 + rng() * 0.1) * fold;
    const p: V3 = [
      0.02 + Math.sin(phi) * Math.cos(theta) * 1.18 * jitter,
      0.06 + Math.cos(phi) * 0.86 * jitter,
      Math.sin(phi) * Math.sin(theta) * 0.7 * jitter,
    ];
    return { p, kind: "shell" };
  }
  if (roll < 0.72) {
    const t = rng();
    const c = ccPoint(t);
    return { p: [c[0] + (rng() - 0.5) * 0.08, c[1] + (rng() - 0.5) * 0.06, (rng() - 0.5) * 0.28], kind: "cc" };
  }
  if (roll < 0.86) {
    const theta = rng() * Math.PI * 2;
    const phi = Math.acos(2 * rng() - 1);
    const j = 0.75 + rng() * 0.28;
    return {
      p: [-0.78 + Math.sin(phi) * Math.cos(theta) * 0.5 * j, -0.58 + Math.cos(phi) * 0.4 * j, Math.sin(phi) * Math.sin(theta) * 0.52 * j],
      kind: "cere",
    };
  }
  if (roll < 0.93) {
    const y = -0.2 - rng() * 1.15;
    const ang = rng() * Math.PI * 2;
    const rad = rng() * 0.14;
    return { p: [stemCenter(y) + Math.cos(ang) * rad, y, Math.sin(ang) * rad], kind: "stem" };
  }
  return { p: [(rng() - 0.45) * 1.3, (rng() - 0.4) * 1.1, (rng() - 0.5) * 0.7], kind: "deep" };
}

export function buildTracts(count: number, seed = 7): Tract {
  const rng = mulberry32(seed);
  const pos: number[] = [];
  const col: number[] = [];
  const seeds: number[] = [];
  let made = 0;
  let guard = 0;
  while (made < count && guard < count * 8) {
    guard += 1;
    const { p, kind } = sampleOrigin(rng);
    if (kind !== "shell" && kind !== "cc" && !insideBrain(p[0], p[1], p[2])) continue;
    const steps = kind === "shell" ? 7 + (rng() * 6) | 0 : 10 + (rng() * 10) | 0;
    const step = kind === "shell" ? 0.032 + rng() * 0.02 : 0.04 + rng() * 0.018;
    const allow = kind === "shell" || kind === "cere" ? 4 : 1;
    const pts = traceFiber(rng, p, steps, step, allow);
    if (pts.length < 3) continue;
    const s = rng();
    for (let i = 1; i < pts.length; i++) {
      const dir = norm([
        pts[i][0] - pts[i - 1][0],
        pts[i][1] - pts[i - 1][1],
        pts[i][2] - pts[i - 1][2],
      ]);
      const mid: V3 = [
        (pts[i][0] + pts[i - 1][0]) * 0.5,
        (pts[i][1] + pts[i - 1][1]) * 0.5,
        (pts[i][2] + pts[i - 1][2]) * 0.5,
      ];
      const c = paint(dir, mid[0], mid[1], mid[2], s);
      pushSeg(pos, col, seeds, pts[i - 1], pts[i], c, s);
      if (kind === "cc" || kind === "shell") {
        const off = 0.012;
        const a: V3 = [pts[i - 1][0], pts[i - 1][1], pts[i - 1][2] + off];
        const b: V3 = [pts[i][0], pts[i][1], pts[i][2] + off];
        pushSeg(pos, col, seeds, a, b, c, s + 0.17);
      }
    }
    made += 1;
  }
  return {
    positions: new Float32Array(pos),
    colors: new Float32Array(col),
    seeds: new Float32Array(seeds),
  };
}

export function regionAt(x: number, y: number, z: number): RegionId | null {
  if (inCerebellum(x, y, z) && !inCerebrum(x, y, z)) return "cerebellum";
  if (inStem(x, y, z) && !inCerebrum(x, y, z)) return "stem";
  if (!inCerebrum(x, y, z) && !inStem(x, y, z)) return null;
  if (Math.abs(x) < 0.22 && Math.abs(y - 0.02) < 0.2 && Math.abs(z) < 0.22) return "thalamus";
  if (y < 0.02 && x > -0.15) return "temporal";
  if (x > 0.32) return "frontal";
  if (x < -0.38) return "occipital";
  if (y > 0.15) return "parietal";
  if (inStem(x, y, z)) return "stem";
  return x > 0 ? "frontal" : "parietal";
}

function randomIn(rng: () => number, pred: (x: number, y: number, z: number) => boolean, box: V3): V3 | null {
  for (let i = 0; i < 80; i++) {
    const p: V3 = [(rng() - 0.5) * box[0], (rng() - 0.5) * box[1], (rng() - 0.5) * box[2]];
    if (pred(p[0], p[1], p[2])) return p;
  }
  return null;
}

const REGION_BOX: Record<RegionId, { pred: (x: number, y: number, z: number) => boolean; box: V3; n: number }> = {
  frontal: { pred: (x, y, z) => regionAt(x, y, z) === "frontal", box: [2.6, 2.2, 1.6], n: 26 },
  parietal: { pred: (x, y, z) => regionAt(x, y, z) === "parietal", box: [2.6, 2.2, 1.6], n: 22 },
  occipital: { pred: (x, y, z) => regionAt(x, y, z) === "occipital", box: [2.6, 2.2, 1.6], n: 16 },
  temporal: { pred: (x, y, z) => regionAt(x, y, z) === "temporal", box: [2.6, 2.2, 1.6], n: 18 },
  thalamus: { pred: (x, y, z) => regionAt(x, y, z) === "thalamus", box: [1.2, 1.2, 1], n: 14 },
  cerebellum: { pred: (x, y, z) => regionAt(x, y, z) === "cerebellum", box: [2.6, 2.4, 1.6], n: 16 },
  stem: { pred: (x, y, z) => regionAt(x, y, z) === "stem", box: [1.4, 3.2, 1], n: 12 },
};

function traceAxon(a: V3, b: V3): Float32Array {
  const steps = 12;
  const pts: number[] = [a[0], a[1], a[2]];
  let p: V3 = [a[0], a[1], a[2]];
  for (let i = 0; i < steps; i++) {
    const t = (i + 1) / (steps + 1);
    const to = norm([b[0] - p[0], b[1] - p[1], b[2] - p[2]]);
    const f = flowAt(p[0], p[1], p[2]);
    const blend = 0.42 + t * 0.45;
    const d = norm([
      f[0] * (1 - blend) + to[0] * blend,
      f[1] * (1 - blend) + to[1] * blend,
      f[2] * (1 - blend) + to[2] * blend,
    ]);
    const hop = len3(b[0] - p[0], b[1] - p[1], b[2] - p[2]) * 0.45;
    p = [p[0] + d[0] * hop, p[1] + d[1] * hop, p[2] + d[2] * hop];
    pts.push(p[0], p[1], p[2]);
  }
  pts.push(b[0], b[1], b[2]);
  return new Float32Array(pts);
}

export function buildNetwork(phone: boolean, seed = 11): { neurons: Neuron[]; axons: Axon[] } {
  const rng = mulberry32(seed);
  const scale = phone ? 0.72 : 1;
  const neurons: Neuron[] = [];
  for (const info of REGIONS) {
    const spec = REGION_BOX[info.id];
    const target = Math.max(8, Math.round(spec.n * scale));
    for (let i = 0; i < target; i++) {
      const p = randomIn(rng, spec.pred, spec.box);
      if (!p) continue;
      neurons.push({ region: info.id, x: p[0], y: p[1], z: p[2] });
    }
  }
  const byRegion = new Map<RegionId, number[]>();
  neurons.forEach((n, i) => {
    const list = byRegion.get(n.region) ?? [];
    list.push(i);
    byRegion.set(n.region, list);
  });
  const axons: Axon[] = [];
  const link = (i: number, j: number) => {
    if (i === j) return;
    const a = neurons[i];
    const b = neurons[j];
    const pts = traceAxon([a.x, a.y, a.z], [b.x, b.y, b.z]);
    let length = 0;
    for (let k = 3; k < pts.length; k += 3) {
      length += len3(pts[k] - pts[k - 3], pts[k + 1] - pts[k - 2], pts[k + 2] - pts[k - 1]);
    }
    axons.push({ from: i, to: j, pts, length: Math.max(0.2, length) });
  };
  const pick = (region: RegionId) => {
    const list = byRegion.get(region);
    if (!list || list.length === 0) return -1;
    return list[(rng() * list.length) | 0];
  };
  for (const [region, list] of byRegion) {
    for (const i of list) {
      const mates = list.filter((j) => j !== i);
      mates.sort((a, b) => {
        const da = len3(neurons[a].x - neurons[i].x, neurons[a].y - neurons[i].y, neurons[a].z - neurons[i].z);
        const db = len3(neurons[b].x - neurons[i].x, neurons[b].y - neurons[i].y, neurons[b].z - neurons[i].z);
        return da - db;
      });
      if (mates[0] !== undefined) link(i, mates[0]);
      if (mates[1] !== undefined && rng() < 0.65) link(i, mates[1]);
    }
    void region;
  }
  const cortical: RegionId[] = ["frontal", "parietal", "occipital", "temporal"];
  for (let k = 0; k < (phone ? 28 : 42); k++) {
    const a = pick("thalamus");
    const b = pick(cortical[(rng() * cortical.length) | 0]);
    if (a >= 0 && b >= 0) link(a, b);
  }
  for (let k = 0; k < (phone ? 10 : 16); k++) {
    const a = pick("stem");
    const b = pick(rng() < 0.5 ? "thalamus" : "cerebellum");
    if (a >= 0 && b >= 0) link(a, b);
  }
  for (let k = 0; k < (phone ? 12 : 18); k++) {
    const a = pick("frontal");
    const b = pick(rng() < 0.5 ? "parietal" : "temporal");
    if (a >= 0 && b >= 0) link(a, b);
  }
  return { neurons, axons };
}
