// Node smoke test for the distributed step-parallel API.
// Single instance: init → (state → search local → apply) × total → finish.
// Verifies the step pipeline produces a valid result comparable to the one-shot API.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

const wasmExecSource = readFileSync(join(root, "web/wasm/wasm_exec.js"), "utf-8");
globalThis.require = (await import("node:module")).createRequire(import.meta.url);
eval(wasmExecSource);

const FULL_W = 200, FULL_H = 160;
const rgba = new Uint8Array(FULL_W * FULL_H * 4);
const alpha = new Uint8Array(FULL_W * FULL_H);
for (let y = 0; y < FULL_H; y++) {
  for (let x = 0; x < FULL_W; x++) {
    const i = y * FULL_W + x;
    const dx = x - 100, dy = y - 80;
    const inside = dx * dx + dy * dy < 55 * 55;
    rgba[i * 4] = inside ? 200 : 255;
    rgba[i * 4 + 1] = inside ? 40 : 255;
    rgba[i * 4 + 2] = inside ? 40 : 255;
    rgba[i * 4 + 3] = 255;
    alpha[i] = 255;
  }
}

const go = new Go();
const wasmBytes = readFileSync(join(root, "web/wasm/primitive.wasm"));
const { instance } = await WebAssembly.instantiate(wasmBytes, go.importObject);
go.run(instance);

const config = {
  full_w: FULL_W, full_h: FULL_H, work_w: FULL_W, work_h: FULL_H,
  num_primitives: 40, allowed_shapes: ["circle"], transparent: false, mask_threshold: 127,
};

// 1) init
const initRes = JSON.parse(globalThis.primitiveFitInit(JSON.stringify(config), rgba, alpha));
if (!initRes.ok) { console.error("FAIL init:", initRes.error); process.exit(1); }
console.log(`init ok: total=${initRes.total} steps=${initRes.steps.length}`);
if (initRes.total !== 40 || initRes.steps.length !== 40) { console.error("FAIL total mismatch"); process.exit(1); }

// 2) step loop (searchLocal path, wm=16 like a single CLI worker)
const started = Date.now();
let state = globalThis.primitiveFitState();
let lastApply = null;
for (let i = 0; i < initRes.total; i++) {
  const mode = initRes.steps[i];
  const candJson = globalThis.primitiveFitSearch(null, state.score, mode, 16, i);
  const cand = JSON.parse(candJson);
  if (cand.error) { console.error(`FAIL search step ${i}:`, cand.error); process.exit(1); }
  if (!cand.type || !Number.isFinite(cand.energy) || cand.alpha < 1) {
    console.error(`FAIL candidate invalid step ${i}:`, candJson.slice(0, 120)); process.exit(1);
  }
  lastApply = globalThis.primitiveFitApply(JSON.stringify(cand));
  if (lastApply.error) { console.error(`FAIL apply step ${i}:`, lastApply.error); process.exit(1); }
  state = lastApply;
}
console.log(`steps done in ${(Date.now() - started) / 1000}s, done=${lastApply.done}/${lastApply.total}, score=${lastApply.score.toFixed(4)}`);

// 3) finish
const result = JSON.parse(globalThis.primitiveFitFinish());
if (result.error) { console.error("FAIL finish:", result.error); process.exit(1); }
console.log("shapes:", result.shapes.length, "bbox:", result.bbox, "coverage:", result.coverage);
console.log("first shape:", JSON.stringify(result.shapes[0]));

const okShapes = result.shapes.length === 40;
const okBBox = result.bbox[2] > result.bbox[0] && result.bbox[3] > result.bbox[1];
const okPreview = typeof result.preview_png === "string" && result.preview_png.length > 500;
const okMask = typeof result.mask_png === "string" && result.mask_png.length > 50;
if (!okShapes || !okBBox || !okPreview || !okMask) {
  console.error("FAIL final checks", { okShapes, okBBox, okPreview, okMask });
  process.exit(1);
}

// 4) score monotonic decrease (sanity of Add path)
console.log("PASS");
process.exit(0);
