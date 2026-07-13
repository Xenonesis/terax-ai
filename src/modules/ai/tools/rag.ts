import { tool } from "ai";
import { z } from "zod";
import { native } from "../lib/native";
import { type ToolContext } from "./context";
import { vectorStore } from "../lib/vectorStore";
import { checkReadableCanonical } from "../lib/security";

export function buildRagTools(ctx: ToolContext) {
  return {
    semantic_search: tool({
      description:
        "Perform a semantic search across the codebase using AI embeddings. Use this when you are looking for abstract concepts, business logic, or general architecture, and a regex 'grep' would be too brittle. This embeds your query and compares it against files. Keep queries natural like 'where are terminal exit codes handled?'. Limits search to max 200 files for performance.",
      inputSchema: z.object({
        query: z.string().describe("The natural language search query."),
        glob: z
          .string()
          .optional()
          .describe("Optional glob pattern to restrict which files are embedded/searched (e.g. 'src/**/*.ts'). Defaults to 'src/**/*.{ts,tsx,rs,md}'."),
      }),
      execute: async ({ query, glob }) => {
        const root = ctx.getWorkspaceRoot() || ctx.getCwd();
        if (!root) return { error: "No workspace root available." };

        const safety = await checkReadableCanonical(root, native.canonicalize);
        if (!safety.ok) return { error: safety.reason };
        const canonicalRoot = safety.canonical;

        const pattern = glob || "**/*.{ts,tsx,rs,js,jsx,md}";

        try {
          // Find all candidate files
          const globRes = await native.glob({
            pattern,
            root: canonicalRoot,
            maxResults: 200, // Limit to 200 files to avoid burning CPU on giant repos
          });

          if (globRes.hits.length === 0) {
            return { message: "No files found matching the glob pattern." };
          }

          const relPaths = globRes.hits.map(h => h.rel);
          
          // Perform semantic search via vectorStore
          const results = await vectorStore.search(query, canonicalRoot, relPaths, 10);

          return {
            query,
            scanned_files: relPaths.length,
            results: results.map(r => ({
              path: r.path,
              similarity: r.score.toFixed(3),
              preview: r.preview
            }))
          };
        } catch (e) {
          return { error: String(e) };
        }
      },
    }),
  } as const;
}
