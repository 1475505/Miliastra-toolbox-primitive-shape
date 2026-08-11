/* Web Worker: hosts the Go/WebAssembly primitive fitting engine.
 *
 * Protocol (main thread -> worker):
 *   { type: "fit", jobId, config, rgba, alpha }            one-shot (fallback)
 *   { type: "init", jobId, config, rgba, alpha }           distributed: build session
 *   { type: "state", jobId }                               distributed: current canvas
 *   { type: "search", jobId, rgba|null, score, mode, wm, nonce }
 *   { type: "apply", jobId, candidate }
 *   { type: "finish", jobId }
 * Worker -> main thread:
 *   { type: "ready" } / { type: "boot_error", message }
 *   { type: "progress", jobId, done, total }               one-shot progress
 *   { type: "result", jobId, json }                        fit / finish
 *   { type: "initResult", jobId, json }
 *   { type: "stateResult", jobId, rgba, score, done }
 *   { type: "searchResult", jobId, json }
 *   { type: "applyResult", jobId, rgba, score, done, total }
 */
"use strict";

importScripts("/web/wasm/wasm_exec.js");

let activeJobId = null;

globalThis.primitiveFitProgress = function (done, total) {
  postMessage({ type: "progress", jobId: activeJobId, done: done, total: total });
};

const go = new Go();

async function boot() {
  try {
    const response = await fetch("/web/wasm/primitive.wasm");
    if (!response.ok) {
      throw new Error("wasm 下载失败: HTTP " + response.status);
    }
    const bytes = await response.arrayBuffer();
    const module = await WebAssembly.instantiate(bytes, go.importObject);
    // go.run never resolves (the Go program blocks on select{}); do not await.
    go.run(module.instance);
    postMessage({ type: "ready" });
  } catch (err) {
    postMessage({ type: "boot_error", message: String((err && err.message) || err) });
  }
}

function asUint8(view) {
  return view instanceof Uint8Array ? view : null;
}

onmessage = function (event) {
  const data = event.data || {};
  const jobId = data.jobId;
  try {
    switch (data.type) {
      case "fit": {
        activeJobId = jobId;
        try {
          const json = globalThis.primitiveFit(JSON.stringify(data.config), data.rgba, data.alpha);
          postMessage({ type: "result", jobId: jobId, json: json });
        } finally {
          activeJobId = null;
        }
        return;
      }
      case "init": {
        const json = globalThis.primitiveFitInit(JSON.stringify(data.config), data.rgba, data.alpha);
        postMessage({ type: "initResult", jobId: jobId, json: json });
        return;
      }
      case "state": {
        const out = globalThis.primitiveFitState();
        if (out && out.error) {
          postMessage({ type: "stateResult", jobId: jobId, error: String(out.error) });
          return;
        }
        const rgba = asUint8(out.rgba);
        postMessage(
          { type: "stateResult", jobId: jobId, rgba: rgba, score: out.score, done: out.done },
          rgba ? [rgba.buffer] : []
        );
        return;
      }
      case "search": {
        const json = globalThis.primitiveFitSearch(
          asUint8(data.rgba),
          data.score,
          data.mode,
          data.wm,
          data.nonce
        );
        postMessage({ type: "searchResult", jobId: jobId, json: json });
        return;
      }
      case "apply": {
        const out = globalThis.primitiveFitApply(data.candidate);
        if (out && out.error) {
          postMessage({ type: "applyResult", jobId: jobId, error: String(out.error) });
          return;
        }
        const rgba = asUint8(out.rgba);
        postMessage(
          { type: "applyResult", jobId: jobId, rgba: rgba, score: out.score, done: out.done, total: out.total },
          rgba ? [rgba.buffer] : []
        );
        return;
      }
      case "finish": {
        const json = globalThis.primitiveFitFinish();
        postMessage({ type: "result", jobId: jobId, json: json });
        return;
      }
      default:
        return;
    }
  } catch (err) {
    const message = String((err && err.message) || err);
    if (data.type === "init") {
      postMessage({ type: "initResult", jobId: jobId, json: JSON.stringify({ error: message }) });
    } else if (data.type === "search") {
      postMessage({ type: "searchResult", jobId: jobId, json: JSON.stringify({ error: message }) });
    } else if (data.type === "state") {
      postMessage({ type: "stateResult", jobId: jobId, error: message });
    } else if (data.type === "apply") {
      postMessage({ type: "applyResult", jobId: jobId, error: message });
    } else {
      postMessage({ type: "result", jobId: jobId, json: JSON.stringify({ error: message }) });
    }
  }
};

boot();
