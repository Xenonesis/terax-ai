import { pipeline, env } from "@huggingface/transformers";

// Disable local models since we only want to load from HuggingFace Hub, or enable local if we ship the model.
// For a desktop app, it's better to let it download to the OPFS/Cache API first.
env.allowLocalModels = false;
env.useBrowserCache = true;

let extractor: any = null;

// Singleton to load the model
async function loadExtractor() {
  if (!extractor) {
    // all-MiniLM-L6-v2 is small and very fast for local semantic search
    extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      dtype: "q8", // quantized 8-bit for smaller size and speed
    });
  }
  return extractor;
}

self.addEventListener("message", async (event) => {
  const { id, type, payload } = event.data;

  if (type === "EMBED") {
    try {
      const ext = await loadExtractor();
      const texts = Array.isArray(payload) ? payload : [payload];
      
      const output = await ext(texts, {
        pooling: "mean",
        normalize: true,
      });

      self.postMessage({
        id,
        status: "success",
        embeddings: output.tolist(),
      });
    } catch (err: any) {
      self.postMessage({
        id,
        status: "error",
        error: err.message,
      });
    }
  }
});
