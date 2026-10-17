// The Web Worker of index.html: fetches the Depth Anything 3 Small weights (once; the
// Cache API keeps them), runs the network on the photos and sends back the result as a
// run (splat.ply, cameras.json, input views). Messages in: {type: "reconstruct", photos:
// [{name, bytes}], focal35mm}. Out: {type: "progress", text, fraction}, {type: "done", ...},
// {type: "error", message}.
import init, * as da3 from "./pkg/instantsplat_web.js";

const WEIGHTS_URL = "https://huggingface.co/depth-anything/DA3-SMALL/resolve/main/model.safetensors";
const WEIGHTS_CACHE = "depth-anything-3-small";

const progress = (text, fraction) => postMessage({ type: "progress", text, fraction });

// Where each stage of the network sits in its share of the work (the backbone's 12 blocks
// take about two thirds of the time, the depth head most of the rest).
const NETWORK_SHARE = { backboneEnd: 0.65, depthHead: 0.7, cameraHead: 0.98 };
const networkFraction = (text) => {
  const block = text.match(/^backbone block (\d+)\/(\d+)/);
  if (block) return (Number(block[1]) / Number(block[2])) * NETWORK_SHARE.backboneEnd;
  if (text.startsWith("depth head")) return NETWORK_SHARE.depthHead;
  if (text.startsWith("camera head")) return NETWORK_SHARE.cameraHead;
  return 0;
};

// The weights as bytes, from the Cache API when they are there, else downloaded into it.
async function weights() {
  const cache = await caches.open(WEIGHTS_CACHE);
  const cached = await cache.match(WEIGHTS_URL);
  if (cached) {
    progress("weights: cached", 1);
    return new Uint8Array(await cached.arrayBuffer());
  }
  const response = await fetch(WEIGHTS_URL);
  if (!response.ok) throw new Error(`${WEIGHTS_URL}: ${response.status}`);
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
  await cache.put(WEIGHTS_URL, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
  return bytes;
}

let ready = null;
async function prepare() {
  await init();
  const started = performance.now();
  const bytes = await weights();
  const fetched = performance.now();
  progress(`loading ${(bytes.length / 1e6).toFixed(0)} MB of weights…`, 1);
  da3.load_da3(bytes);
  progress(`weights: fetched in ${((fetched - started) / 1000).toFixed(1)} s, loaded in ${((performance.now() - fetched) / 1000).toFixed(1)} s`, 1);
}

onmessage = async ({ data }) => {
  try {
    if (data.type !== "reconstruct") throw new Error(`unknown message ${data.type}`);
    ready ??= prepare();
    await ready;
    for (const { name, bytes } of data.photos) da3.add_input_photo(name, bytes);
    const result = da3.reconstruct(data.focal35mm ?? undefined, (text) => progress(text, networkFraction(text)));
    const views = [];
    for (let i = 0; i < result.view_count(); i++) views.push(result.input_view_png(i));
    const ply = result.splat_ply();
    const message = { type: "done", ply, camerasJson: result.cameras_json(), views, report: result.report() };
    result.free();
    postMessage(message, [ply.buffer, ...views.map((view) => view.buffer)]);
  } catch (error) {
    postMessage({ type: "error", message: error.message ?? String(error) });
  }
};
