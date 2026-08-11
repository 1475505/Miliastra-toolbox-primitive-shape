// Node smoke test for the primitive WASM build.
// Loads web/wasm/primitive.wasm, feeds a synthetic RGBA image, prints the result summary.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

// wasm_exec.js expects global `Go` and (in node) global fs/crypto — it self-detects node.
const wasmExecSource = readFileSync(join(root, "web/wasm/wasm_exec.js"), "utf-8");
globalThis.require = (await import("node:module")).createRequire(import.meta.url);
eval(wasmExecSource);

// decode a PNG into RGBA using the PNG that the Go side would receive.
// Instead of decoding PNG in node, synthesize a simple test image: white bg + red circle.
const FULL_W = 200, FULL_H = 160;
const WORK_W = 200, WORK_H = 160;
const rgba = new Uint8Array(WORK_W * WORK_H * 4);
const alpha = new Uint8Array(WORK_W * WORK_H);
for (let y = 0; y < WORK_H; y++) {
  for (let x = 0; x < WORK_W; x++) {
    const i = y * WORK_W + x;
    const dx = x - 100, dy = y - 80;
    const inside = dx * dx + dy * dy < 55 * 55;
    rgba[i * 4] = inside ? 200 : 255;
    rgba[i * 4 + 1] = inside ? 40 : 255;
    rgba[i * 4 + 2] = inside ? 40 : 255;
    rgba[i * 4 + 3] = 255;
    alpha[i] = 255;
  }
}

let progressCalls = 0;
globalThis.primitiveFitProgress = (done, total) => {
  progressCalls++;
  if (done % 20 === 0 || done === total) console.log(`progress ${done}/${total}`);
};

const go = new Go();
const wasmBytes = readFileSync(join(root, "web/wasm/primitive.wasm"));
const { instance } = await WebAssembly.instantiate(wasmBytes, go.importObject);
go.run(instance);

if (typeof globalThis.primitiveFit !== "function") {
  console.error("FAIL: primitiveFit not exported");
  process.exit(1);
}

const config = {
  full_w: FULL_W, full_h: FULL_H, work_w: WORK_W, work_h: WORK_H,
  num_primitives: 80, allowed_shapes: ["circle"], transparent: false, mask_threshold: 127,
};
const started = Date.now();
const json = globalThis.primitiveFit(JSON.stringify(config), rgba, alpha);
const result = JSON.parse(json);
console.log(`elapsed ${(Date.now() - started) / 1000}s, progress calls: ${progressCalls}`);
if (result.error) {
  console.error("FAIL:", result.error);
  process.exit(1);
}
console.log("shapes:", result.shapes.length);
console.log("first shape:", JSON.stringify(result.shapes[0]));
console.log("bbox:", result.bbox, "coverage:", result.coverage);
console.log("preview png bytes:", result.preview_png.length, "mask png bytes:", (result.mask_png || "").length);

// sanity assertions
const okShapes = result.shapes.length >= 70 && result.shapes.length <= 80;
const c = result.shapes[0];
const okType = c && c.type === "circle" && c.color.startsWith("#") && c.packed_color > 0;
const okBBox = result.bbox[2] > result.bbox[0] && result.bbox[3] > result.bbox[1];
const okMask = typeof result.mask_png === "string" && result.mask_png.length > 50;
if (!okShapes || !okType || !okBBox || !okMask) {
  console.error("FAIL: sanity checks", { okShapes, okType, okBBox, okMask });
  process.exit(1);
}
console.log("PASS");
process.exit(0);
