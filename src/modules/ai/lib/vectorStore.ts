import { LazyStore } from "@tauri-apps/plugin-store";
import { native } from "./native";

export type CachedEmbedding = {
  mtime: number;
  embedding: number[];
  textPreview: string;
};

// Cosine similarity function
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class VectorStore {
  private store: LazyStore;
  private worker: Worker;
  private reqIdCounter = 0;
  private resolvers = new Map<number, { resolve: (res: any) => void; reject: (err: any) => void }>();

  constructor() {
    this.store = new LazyStore("terax-rag-store.json");
    
    // Initialize Web Worker for Transformers.js
    this.worker = new Worker(new URL("./embeddingWorker.ts", import.meta.url), {
      type: "module",
    });

    this.worker.addEventListener("message", (event) => {
      const { id, status, embeddings, error } = event.data;
      const deferred = this.resolvers.get(id);
      if (deferred) {
        if (status === "success") {
          deferred.resolve(embeddings);
        } else {
          deferred.reject(new Error(error));
        }
        this.resolvers.delete(id);
      }
    });
  }

  // Embed a text using the Web Worker
  public async embedText(text: string): Promise<number[][]> {
    const id = ++this.reqIdCounter;
    return new Promise((resolve, reject) => {
      this.resolvers.set(id, { resolve, reject });
      this.worker.postMessage({ id, type: "EMBED", payload: text });
    }) as Promise<number[][]>;
  }

  // Gets or creates an embedding for a file
  public async getFileEmbedding(workspace: string, relPath: string): Promise<CachedEmbedding | null> {
    const absPath = `${workspace}/${relPath}`;
    
    // We don't have mtime from glob, so we stat the file by doing readDir on its parent
    // Wait, native doesn't expose stat directly, so we just read the file for MVP 
    // and hash it, or we skip caching for now if we can't easily stat.
    // For MVP, let's just cache by content hash or full text.
    
    let content = "";
    try {
      const readRes = await native.readFile(absPath);
      if (readRes.kind !== "text") return null;
      content = readRes.content;
    } catch {
      return null;
    }

    // Hash the content to use as an mtime-substitute (simple string length + sample)
    const contentHash = content.length + content.slice(0, 100);
    const cacheKey = `embed:${workspace}:${relPath}`;

    const cached = await this.store.get<CachedEmbedding>(cacheKey);
    // Simple cache validation based on contentHash (stored as string in mtime field for hacky MVP)
    if (cached && String(cached.mtime) === contentHash) {
      return cached;
    }

    // Need to embed
    // We only embed the first 2000 chars for speed in MVP
    const textPreview = content.slice(0, 2000);
    try {
      const matrix = await this.embedText(textPreview);
      // Transformers pipeline returns nested array [ [ vector ] ]
      const embedding = matrix[0]; 
      
      const newCache: CachedEmbedding = {
        mtime: contentHash as any,
        embedding,
        textPreview: textPreview.slice(0, 200) + "..."
      };
      
      await this.store.set(cacheKey, newCache);
      await this.store.save(); // Persist store
      return newCache;
    } catch (e) {
      console.warn("Failed to embed", relPath, e);
      return null;
    }
  }

  // Perform a semantic search across a list of files
  public async search(query: string, workspace: string, relPaths: string[], topK = 5) {
    const queryEmbedMatrix = await this.embedText(query);
    const queryEmbed = queryEmbedMatrix[0];

    const results: Array<{ path: string; score: number; preview: string }> = [];

    for (const relPath of relPaths) {
      const fileData = await this.getFileEmbedding(workspace, relPath);
      if (fileData && fileData.embedding) {
        const score = cosineSimilarity(queryEmbed, fileData.embedding);
        results.push({ path: relPath, score, preview: fileData.textPreview });
      }
    }

    return results.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}

// Singleton
export const vectorStore = new VectorStore();
