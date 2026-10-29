import * as THREE from "three";
import "./style.css";
import { buildTracts, REGIONS, type RegionId } from "./anatomy";
import { buildLayout, JOINT_KO, LAYERS, type NeuronPt, type Rec } from "./net";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function srgbToLinear(c: number) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}
function hexLin(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [srgbToLinear(((n >> 16) & 255) / 255), srgbToLinear(((n >> 8) & 255) / 255), srgbToLinear((n & 255) / 255)];
}
const REGION_COLOR = Object.fromEntries(REGIONS.map((r) => [r.id, r.color])) as Record<RegionId, string>;
const REGION_KO = Object.fromEntries(REGIONS.map((r) => [r.id, r.label])) as Record<RegionId, string>;

function circleTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(32, 32, 1, 32, 32, 31);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.3, "rgba(255,255,255,0.8)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

function commandText(c: number[]) {
  const [vx, , wz] = c;
  if (vx === 0 && wz === 0) return "제자리에 서 있기";
  const parts: string[] = [];
  if (vx > 0) parts.push(`앞으로 ${vx.toFixed(2)} m/s`);
  if (wz > 0) parts.push(`왼쪽으로 돌기 ${wz.toFixed(1)} rad/s`);
  if (wz < 0) parts.push(`오른쪽으로 돌기 ${(-wz).toFixed(1)} rad/s`);
  if (vx === 0 && wz !== 0) parts[0] = "제자리에서 " + parts[0];
  return parts.join(" + ");
}

function percentile(values: number[], p: number) {
  if (!values.length) return 1;
  const s = values.slice().sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))] || 1;
}

async function main() {
  const res = await fetch("data/activity.json");
  const rec = (await res.json()) as Rec;
  const N = rec.t.length;
  const fps = rec.meta.fps;
  const duration = N / fps;
  const SC = rec.meta.scale;
  const pts: NeuronPt[] = buildLayout(rec);
  const sizes = rec.meta.layer_sizes;
  const lookup: number[][] = sizes.map((n) => new Array(n).fill(-1));
  pts.forEach((p, i) => (lookup[p.layer][p.idx] = i));

  // brightness scale per layer: hidden = positive ELU part, input/output = magnitude
  const signedLayer = (l: number) => l === 0 || l === 4;
  const scales = sizes.map((_, l) => {
    const vals: number[] = [];
    for (let f = 0; f < N; f += 3) for (const v of rec.L[l][f]) vals.push(signedLayer(l) ? Math.abs(v) : Math.max(0, v));
    return percentile(vals.filter((v) => v > 0), 0.92) / SC;
  });

  // ---------- three.js scene (tract backdrop reused from 신경속) ----------
  const canvas = $<HTMLCanvasElement>("brain");
  const phone = window.matchMedia("(pointer: coarse)").matches;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: !phone, powerPreference: "high-performance", preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, phone ? 1.35 : 1.6));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x000000, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 40);
  let yaw = 0.55, pitch = 0.16, dist = 4.1;
  const target = new THREE.Vector3(-0.05, -0.15, 0);

  const tracts = buildTracts(phone ? 1100 : 2200, 7);
  const tractGeo = new THREE.BufferGeometry();
  tractGeo.setAttribute("position", new THREE.BufferAttribute(tracts.positions, 3));
  tractGeo.setAttribute("aColor", new THREE.BufferAttribute(tracts.colors, 3));
  const tractMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uGain: { value: 0.24 } },
    vertexShader: `attribute vec3 aColor; uniform float uGain; varying vec3 vC;
      void main(){ vC = aColor * uGain; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `varying vec3 vC; void main(){ gl_FragColor = vec4(vC, 0.85); }`,
  });
  scene.add(new THREE.LineSegments(tractGeo, tractMat));

  // neurons
  const P = pts.length;
  const nPos = new Float32Array(P * 3), nCol = new Float32Array(P * 3), nAct = new Float32Array(P), nSign = new Float32Array(P), nBase = new Float32Array(P);
  pts.forEach((p, i) => {
    nPos.set([p.x, p.y, p.z], i * 3);
    nCol.set(hexLin(REGION_COLOR[p.region]), i * 3);
    nBase[i] = p.layer === 4 ? 2.2 : p.layer === 0 ? 1.5 : 1.0;
  });
  const nGeo = new THREE.BufferGeometry();
  nGeo.setAttribute("position", new THREE.BufferAttribute(nPos, 3));
  nGeo.setAttribute("aColor", new THREE.BufferAttribute(nCol, 3));
  nGeo.setAttribute("aAct", new THREE.BufferAttribute(nAct, 1));
  nGeo.setAttribute("aSign", new THREE.BufferAttribute(nSign, 1));
  nGeo.setAttribute("aBase", new THREE.BufferAttribute(nBase, 1));
  const tex = circleTexture();
  const nMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uMap: { value: tex }, uScale: { value: 1 } },
    vertexShader: `attribute vec3 aColor; attribute float aAct; attribute float aSign; attribute float aBase;
      uniform float uScale; varying vec3 vC; varying float vA;
      void main(){
        vec4 mv = modelViewMatrix * vec4(position,1.0);
        float a = clamp(aAct, 0.0, 1.0);
        vec3 hot = aSign < 0.0 ? vec3(0.45,0.85,1.0) : vec3(1.0,0.86,0.55);
        vC = mix(aColor * 0.55, hot, a * 0.75) * (0.35 + 1.5 * a);
        vA = 0.35 + 0.65 * a;
        gl_PointSize = uScale * aBase * (7.0 + 20.0 * a) / -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `uniform sampler2D uMap; varying vec3 vC; varying float vA;
      void main(){ vec4 t = texture2D(uMap, gl_PointCoord); gl_FragColor = vec4(vC * t.a, t.a * vA); }`,
  });
  scene.add(new THREE.Points(nGeo, nMat));

  // strongest-weight connections of the current frame (|w * a_src| top-K per layer pair)
  const K = rec.meta.topk_edges_per_layer;
  const SEG = 10;
  const E = 4 * K;
  const ePos = new Float32Array(E * SEG * 2 * 3), eCol = new Float32Array(E * SEG * 2 * 3);
  const eGeo = new THREE.BufferGeometry();
  eGeo.setAttribute("position", new THREE.BufferAttribute(ePos, 3));
  eGeo.setAttribute("color", new THREE.BufferAttribute(eCol, 3));
  const eMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.9 });
  scene.add(new THREE.LineSegments(eGeo, eMat));

  const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3(), Q = new THREE.Vector3();
  function ctrlOf(a: THREE.Vector3, b: THREE.Vector3, out: THREE.Vector3) {
    out.addVectors(a, b).multiplyScalar(0.5 * 0.55);
    out.y += 0.1;
    return out;
  }
  function bez(a: THREE.Vector3, c: THREE.Vector3, b: THREE.Vector3, u: number, out: THREE.Vector3) {
    const v = 1 - u;
    out.set(v * v * a.x + 2 * v * u * c.x + u * u * b.x, v * v * a.y + 2 * v * u * c.y + u * u * b.y, v * v * a.z + 2 * v * u * c.z + u * u * b.z);
    return out;
  }
  const posOf = (layer: number, idx: number, out: THREE.Vector3) => {
    const i = lookup[layer][idx];
    return out.set(nPos[i * 3], nPos[i * 3 + 1], nPos[i * 3 + 2]);
  };

  function writeEdges(f: number) {
    let o = 0;
    for (let li = 0; li < 4; li++) {
      const list = rec.edges[li][f];
      list.forEach(([s, d, sg], rank) => {
        posOf(li, s, A); posOf(li + 1, d, B); ctrlOf(A, B, C);
        const w = 0.55 * (1 - rank / (K + 4));
        const col = sg >= 0 ? [0.95 * w, 0.62 * w, 0.25 * w] : [0.3 * w, 0.7 * w, 0.95 * w];
        for (let k = 0; k < SEG; k++) {
          bez(A, C, B, k / SEG, Q); ePos.set([Q.x, Q.y, Q.z], o * 3); eCol.set(col, o * 3); o++;
          bez(A, C, B, (k + 1) / SEG, Q); ePos.set([Q.x, Q.y, Q.z], o * 3); eCol.set(col, o * 3); o++;
        }
      });
    }
    eGeo.setDrawRange(0, o);
    eGeo.attributes.position.needsUpdate = true;
    eGeo.attributes.color.needsUpdate = true;
  }

  // pulses travelling along the strongest connections (input -> output cascade)
  type Pulse = { a: THREE.Vector3; b: THREE.Vector3; c: THREE.Vector3; u: number; sg: number };
  const pulses: Pulse[] = [];
  const MAXP = 700;
  const pPos = new Float32Array(MAXP * 3), pCol = new Float32Array(MAXP * 3);
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
  pGeo.setAttribute("color", new THREE.BufferAttribute(pCol, 3));
  const pMat = new THREE.PointsMaterial({ size: phone ? 0.055 : 0.045, map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, vertexColors: true });
  scene.add(new THREE.Points(pGeo, pMat));
  function spawn(f: number) {
    for (let li = 0; li < 4; li++) {
      rec.edges[li][f].slice(0, 6).forEach(([s, d, sg]) => {
        const a = posOf(li, s, new THREE.Vector3()), b = posOf(li + 1, d, new THREE.Vector3());
        pulses.push({ a, b, c: ctrlOf(a, b, new THREE.Vector3()), u: -li * 0.22, sg });
      });
    }
    while (pulses.length > MAXP) pulses.shift();
  }
  function stepPulses(dt: number) {
    let n = 0;
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      p.u += dt / 0.42;
      if (p.u >= 1) { pulses.splice(i, 1); continue; }
      if (p.u < 0) continue;
      bez(p.a, p.c, p.b, p.u, Q);
      pPos.set([Q.x, Q.y, Q.z], n * 3);
      const h = 0.7 + 0.3 * Math.sin(p.u * 10);
      pCol.set(p.sg >= 0 ? [h, 0.8 * h, 0.4 * h] : [0.4 * h, 0.85 * h, h], n * 3);
      n++;
    }
    pGeo.setDrawRange(0, n);
    pGeo.attributes.position.needsUpdate = true;
    pGeo.attributes.color.needsUpdate = true;
  }

  // ---------- region labels ----------
  const labelsEl = $("labels");
  type Lab = { el: HTMLElement; p: THREE.Vector3 };
  const labs: Lab[] = [];
  const centroid = (filter: (p: NeuronPt) => boolean, dy = 0) => {
    const v = new THREE.Vector3(); let n = 0;
    pts.forEach((p) => { if (filter(p)) { v.x += p.x; v.y += p.y; v.z += p.z; n++; } });
    v.multiplyScalar(1 / Math.max(1, n)); v.y += dy; return v;
  };
  const addLab = (title: string, sub: string, color: string, p: THREE.Vector3) => {
    const el = document.createElement("div");
    el.className = "rl";
    el.style.color = color;
    el.innerHTML = `${title}<small>${sub}</small>`;
    labelsEl.appendChild(el);
    labs.push({ el, p });
  };
  addLab("시상 · 입력", "몸 기울기·회전·명령", REGION_COLOR.thalamus, centroid((p) => p.layer === 0 && p.region === "thalamus", 0.05));
  addLab("두정엽 · 입력", "관절 각도·속도", REGION_COLOR.parietal, centroid((p) => p.layer === 0 && p.region === "parietal", 0.25));
  addLab("후두엽 · 입력", "직전 출력(동작 기억)", REGION_COLOR.occipital, centroid((p) => p.layer === 0 && p.region === "occipital", 0.1));
  addLab("전두엽", "은닉층 1 · 뉴런 512개", REGION_COLOR.frontal, centroid((p) => p.layer === 1, 0.25));
  addLab("측두엽", "은닉층 2 · 뉴런 256개", REGION_COLOR.temporal, centroid((p) => p.layer === 2, -0.18));
  addLab("소뇌", "은닉층 3 · 뉴런 128개", REGION_COLOR.cerebellum, centroid((p) => p.layer === 3, -0.25));
  addLab("뇌간 · 출력", "관절 14개 목표 각도", REGION_COLOR.stem, new THREE.Vector3(-0.05, -1.22, 0.15));

  // ---------- side panel ----------
  const layersEl = $("layers");
  const layerMeters: HTMLElement[] = [];
  LAYERS.forEach((L) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot" style="background:${L.color}"></span><span>${L.sub} <span style="color:#8d8e84">· ${L.name}</span></span><span class="meter"><i style="background:${L.color}"></i></span>`;
    layersEl.appendChild(li);
    layerMeters.push(li.querySelector("i")!);
  });
  const jointsEl = $("joints");
  const jointBars: { bar: HTMLElement; num: HTMLElement }[] = [];
  rec.meta.act_joints.forEach((j, i) => {
    if (i === 0 || i === 5 || i === 9) {
      const g = document.createElement("li");
      g.className = "grp";
      g.textContent = i === 0 ? "왼쪽 다리" : i === 5 ? "목·머리" : "오른쪽 다리";
      jointsEl.appendChild(g);
    }
    const li = document.createElement("li");
    li.innerHTML = `<span>${JOINT_KO[j] ?? j}</span><span class="sb"><i></i></span><span class="num">0.00</span>`;
    jointsEl.appendChild(li);
    jointBars.push({ bar: li.querySelector("i")!, num: li.querySelector(".num")! });
  });
  // timeline segment labels
  const segsEl = $("segs");
  const segNames = (c: number[]) => (c[1] === 0 && c[3] === 0 && c[2] === 0 ? "서기" : c[3] > 0 ? "왼쪽으로 돌기" : c[3] < 0 ? "오른쪽으로 돌기" : "직진");
  let acc = 0;
  for (const seg of rec.meta.plan) {
    const s = document.createElement("span");
    s.style.left = `${(acc / duration) * 100}%`;
    s.style.width = `${(seg[0] / duration) * 100}%`;
    s.textContent = segNames(seg);
    segsEl.appendChild(s);
    acc += seg[0];
  }

  // ---------- playback ----------
  const vid = $<HTMLVideoElement>("vid");
  const scrub = $<HTMLInputElement>("scrub");
  const playBtn = $<HTMLButtonElement>("play");
  const speedBtn = $<HTMLButtonElement>("speed");
  scrub.max = String(N - 1);
  let t = 0, playing = true, speed = 1, lastFrame = -1, autoYaw = true;
  const params = new URLSearchParams(location.search);
  if (params.has("t")) t = Math.max(0, Math.min(duration - 0.01, parseFloat(params.get("t")!) || 0));
  if (params.has("paused")) playing = false;

  const setPlaying = (p: boolean) => {
    playing = p;
    playBtn.textContent = p ? "❚❚" : "▶";
    if (p) { vid.playbackRate = speed; vid.play().catch(() => {}); } else { vid.pause(); vid.currentTime = t; }
  };
  playBtn.onclick = () => setPlaying(!playing);
  speedBtn.onclick = () => {
    speed = speed === 1 ? 0.5 : speed === 0.5 ? 0.25 : 1;
    speedBtn.textContent = `${speed}×`;
    vid.playbackRate = speed;
  };
  scrub.oninput = () => { t = (Number(scrub.value) + 0.5) / fps; vid.currentTime = t; pulses.length = 0; };
  window.addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); } });

  function applyFrame(tt: number) {
    const ff = Math.max(0, Math.min(N - 1, tt * fps));
    const f0 = Math.floor(ff), f1 = Math.min(N - 1, f0 + 1), w = ff - f0;
    const means = [0, 0, 0, 0, 0];
    for (let l = 0; l < 5; l++) {
      const a0 = rec.L[l][f0], a1 = rec.L[l][f1];
      const inv = 1 / (scales[l] * SC);
      for (let i = 0; i < a0.length; i++) {
        const v = a0[i] * (1 - w) + a1[i] * w;
        const k = lookup[l][i];
        const m = signedLayer(l) ? Math.abs(v) : Math.max(0, v);
        const b = Math.min(1, m * inv);
        nAct[k] = b; nSign[k] = v < 0 ? -1 : 1;
        means[l] += b;
      }
      means[l] /= a0.length;
    }
    nGeo.attributes.aAct.needsUpdate = true;
    nGeo.attributes.aSign.needsUpdate = true;
    const f = Math.min(N - 1, Math.floor(ff));
    if (f !== lastFrame) {
      writeEdges(f);
      if (playing) spawn(f);
      lastFrame = f;
      scrub.value = String(f);
      $("cmd").textContent = commandText(rec.cmd[f]);
      const b = rec.base[f];
      $("meas").textContent = `실제 속도: 앞 ${b[7].toFixed(2)} m/s · 회전 ${b[8].toFixed(2)} rad/s`;
      $("time").textContent = `${rec.t[f].toFixed(1)}s`;
      const out = rec.L[4][f];
      out.forEach((v, i) => {
        const a = v / SC;
        const pct = Math.min(1, Math.abs(a)) * 50;
        const bar = jointBars[i].bar;
        bar.style.left = a >= 0 ? "50%" : `${50 - pct}%`;
        bar.style.width = `${pct}%`;
        bar.style.background = a >= 0 ? "var(--pos)" : "var(--neg)";
        jointBars[i].num.textContent = (a >= 0 ? "+" : "") + a.toFixed(2);
      });
      means.forEach((m, l) => (layerMeters[l].style.width = `${Math.min(100, m * 160)}%`));
    }
  }

  // ---------- camera controls ----------
  const stage = $("stage");
  let drag: { x: number; y: number } | null = null;
  stage.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; autoYaw = false; stage.setPointerCapture(e.pointerId); });
  stage.addEventListener("pointerup", () => (drag = null));
  stage.addEventListener("pointercancel", () => (drag = null));
  stage.addEventListener("wheel", (e) => { e.preventDefault(); dist = Math.max(1.8, Math.min(8, dist * (e.deltaY > 0 ? 1.08 : 0.93))); }, { passive: false });
  const tip = $("tip");
  const ray = new THREE.Raycaster();
  const ptr = new THREE.Vector2();
  stage.addEventListener("pointermove", (e) => {
    if (drag) {
      yaw -= (e.clientX - drag.x) * 0.0055;
      pitch = Math.max(-1.1, Math.min(1.1, pitch + (e.clientY - drag.y) * 0.004));
      drag = { x: e.clientX, y: e.clientY };
      tip.hidden = true;
      return;
    }
    const r = canvas.getBoundingClientRect();
    ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    let best = 0.035 * (dist / 4), bi = -1;
    for (let i = 0; i < P; i++) {
      Q.set(nPos[i * 3], nPos[i * 3 + 1], nPos[i * 3 + 2]);
      const d = ray.ray.distanceToPoint(Q);
      if (d < best) { best = d; bi = i; }
    }
    if (bi < 0) { tip.hidden = true; return; }
    const p = pts[bi];
    const f = Math.max(0, lastFrame);
    const v = rec.L[p.layer][f][p.idx] / SC;
    tip.hidden = false;
    tip.style.left = `${e.clientX - r.left + 14}px`;
    tip.style.top = `${e.clientY - r.top + 10}px`;
    tip.innerHTML = `<b style="color:${REGION_COLOR[p.region]}">${REGION_KO[p.region]}</b> · ${p.label}<br><span style="font-family:var(--mono)">값 ${v >= 0 ? "+" : ""}${v.toFixed(3)}</span>`;
  });

  function resize() {
    const w = stage.clientWidth || 1, h = stage.clientHeight || 1;
    camera.aspect = w / h;
    // keep the brain clear of the side panel on wide screens
    camera.setViewOffset(w, h, w > 860 ? 160 : 0, w > 860 ? 0 : h * 0.12, w, h);
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    nMat.uniforms.uScale.value = renderer.getPixelRatio() * (h / 900) * 4.2;
    if (w < 860) dist = Math.max(dist, Math.min(8, 4.1 / Math.max(0.5, camera.aspect * 1.25)));
  }
  new ResizeObserver(resize).observe(stage);
  resize();

  const tmp = new THREE.Vector3();
  let last = performance.now();
  function frame(now: number) {
    const dt = Math.max(0, Math.min(0.05, (now - last) / 1000));
    last = now;
    if (playing) {
      t += dt * speed;
      if (t >= duration) { t = 0; pulses.length = 0; vid.currentTime = 0; }
      if (!vid.paused && Math.abs(vid.currentTime - t) > 0.15) vid.currentTime = t;
      if (vid.paused) vid.play().catch(() => {});
      if (autoYaw) yaw += dt * 0.08;
    }
    stepPulses(playing ? dt * speed : 0);
    applyFrame(t);
    const cp = Math.cos(pitch);
    camera.position.set(target.x + Math.sin(yaw) * cp * dist, target.y + Math.sin(pitch) * dist, target.z + Math.cos(yaw) * cp * dist);
    camera.lookAt(target);
    renderer.render(scene, camera);
    const w = stage.clientWidth, h = stage.clientHeight;
    for (const L of labs) {
      tmp.copy(L.p).project(camera);
      const vis = tmp.z < 1;
      L.el.style.display = vis ? "" : "none";
      L.el.style.left = `${(tmp.x * 0.5 + 0.5) * w}px`;
      L.el.style.top = `${(-tmp.y * 0.5 + 0.5) * h}px`;
    }
    requestAnimationFrame(frame);
  }
  vid.currentTime = t;
  setPlaying(playing);
  $("loading").remove();
  (window as unknown as { __ready: boolean }).__ready = true;
  requestAnimationFrame(frame);
}

main().catch((e) => {
  $("loading").textContent = "불러오기 실패: " + String(e);
});
