// The Web Worker of index.html: fetches the weights of a Depth Anything 3 variant (once
// per variant; the Cache API keeps them), runs the network on the photos and sends back
// the result as a run (splat.ply, cameras.json, input views). Messages in, each answered
// by one message out after any number of {type: "progress", text, fraction}; `variant` is
// a MODELS key and every answer repeats the one it is about:
//   {type: "status", variant} -> {type: "status", variant, cached}: weights in the cache?
//   {type: "load", variant} -> {type: "ready", variant}: fetch (or take from the cache) and
//     load the weights, replacing the variant loaded before
//   {type: "reconstruct", photos: [{name, bytes}], focal35mm} -> {type: "done", variant, ...}
// Any failure answers {type: "error", message}. Reconstruct needs a load first.
import init, * as da3 from "./pkg/instantsplat_web.js";

// The variants, by the name `load_da3` takes (the CLI's `--network` names). Not the Giant:
// its 5.4 GB of f32 weights do not fit wasm32's 4 GB of memory.
const MODELS = {
  "da3-small": {
    url: "https://huggingface.co/depth-anything/DA3-SMALL/resolve/main/model.safetensors",
    cacheName: "depth-anything-3-small",
    label: "Depth Anything 3 Small",
  },
  "da3-base": {
    url: "https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors",
    cacheName: "depth-anything-3-base",
    label: "Depth Anything 3 Base",
  },
  "da3-large": {
    url: "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors",
    cacheName: "depth-anything-3-large",
    label: "Depth Anything 3 Large",
  },
};
function model(variant) {
  if (!Object.hasOwn(MODELS, variant)) throw new Error(`unknown model ${variant}`);
  return MODELS[variant];
}

const progress = (text, fraction) => postMessage({ type: "progress", text, fraction });

// Where each stage of the network sits in its share of the work (the backbone's 12 blocks
// take about two thirds of the time, the depth head most of the rest).
const NETWORK_SHARE = { backboneEnd: 0.65, depthHead: 0.65, cameraHead: 0.97 };
const networkFraction = (text) => {
  const block = text.match(/^backbone block (\d+)\/(\d+)/);
  // Stages are announced as they start: block i of N running means i - 1 are done.
  if (block) return ((Number(block[1]) - 1) / Number(block[2])) * NETWORK_SHARE.backboneEnd;
  if (text.startsWith("depth head")) return NETWORK_SHARE.depthHead;
  if (text.startsWith("camera head")) return NETWORK_SHARE.cameraHead;
  return 0;
};

// A variant's weights as bytes, from the Cache API when they are there, else downloaded
// into it.
async function weights({ url, cacheName }) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(url);
  if (cached) {
    progress("weights: cached", 1);
    return new Uint8Array(await cached.arrayBuffer());
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("Content-Length"));
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    progress(`downloading weights: ${(received / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ""} MB`, total ? received / total : 0);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  await cache.put(url, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
  return bytes;
}

async function weightsCached({ url, cacheName }) {
  const cache = await caches.open(cacheName);
  return (await cache.match(url)) !== undefined;
}

// The variant whose model is loaded, if any.
let loaded = null;
async function load(variant) {
  if (loaded === variant) return;
  await init();
  const started = performance.now();
  const { label, ...files } = model(variant);
  const bytes = await weights(files);
  const fetched = performance.now();
  progress(`loading ${label}: ${(bytes.length / 1e6).toFixed(0)} MB of weights…`, 1);
  da3.load_da3(variant, bytes);
  loaded = variant;
  progress(`weights: fetched in ${((fetched - started) / 1000).toFixed(1)} s, loaded in ${((performance.now() - fetched) / 1000).toFixed(1)} s`, 1);
}

const HANDLERS = {
  status: async ({ variant }) => postMessage({ type: "status", variant, cached: await weightsCached(model(variant)) }),
  load: async ({ variant }) => { await load(variant); postMessage({ type: "ready", variant }); },
  reconstruct: async (data) => {
    if (loaded === null) throw new Error("no model is loaded: send a load message first");
    for (const { name, bytes } of data.photos) da3.add_input_photo(name, bytes);
    const result = da3.reconstruct(data.focal35mm ?? undefined, (text) => progress(text, networkFraction(text)));
    const views = [];
    for (let i = 0; i < result.view_count(); i++) views.push(result.input_view_png(i));
    const ply = result.splat_ply();
    const message = { type: "done", variant: loaded, ply, camerasJson: result.cameras_json(), views, report: result.report() };
    result.free();
    postMessage(message, [ply.buffer, ...views.map((view) => view.buffer)]);
  },
};

onmessage = async ({ data }) => {
  try {
    const handler = HANDLERS[data.type];
    if (!handler) throw new Error(`unknown message ${data.type}`);
    await handler(data);
  } catch (error) {
    postMessage({ type: "error", message: error.message ?? String(error) });
  }
};
