import { tool } from "ai";
import { z } from "zod";
import { native } from "../lib/native";
import { resolvePath, type ToolContext } from "./context";
import { checkWritableCanonical } from "../lib/security";

export function buildMemoryTools(ctx: ToolContext) {
  return {
    update_project_memory: tool({
      description:
        "Update the project's living memory (TERAX.md). Use this when the user tells you a rule, preference, or architectural decision that should be remembered across all future sessions in this workspace. The new rule will be appended to the file. Keep the rule concise and actionable.",
      inputSchema: z.object({
        rule: z
          .string()
          .describe("The new rule or preference to remember, formatted as a markdown bullet point."),
      }),
      needsApproval: true,
      execute: async ({ rule }) => {
        const root = ctx.getWorkspaceRoot();
        if (!root) return { error: "No workspace root available to save memory." };

        const memoryPath = resolvePath("TERAX.md", root);
        
        const safety = await checkWritableCanonical(memoryPath, native.canonicalize);
        if (!safety.ok) return { error: safety.reason };
        const canonical = safety.canonical;

        try {
          // Check if it exists
          let content = "";
          try {
            const readRes = await native.readFile(canonical);
            if (readRes.kind === "text") {
              content = readRes.content;
            }
          } catch (e) {
            // File might not exist, that's fine
            content = "# TERAX.md\n\nThis is the living architecture and memory document for this project. The AI agent reads this on every new chat session.\n\n## Project Rules & Memory\n";
          }

          // Ensure it has a rules section
          if (!content.includes("## Project Rules & Memory")) {
            content += "\n## Project Rules & Memory\n";
          }

          // Append rule
          const newRule = rule.startsWith("-") ? rule : `- ${rule}`;
          if (!content.endsWith("\n")) content += "\n";
          content += `${newRule}\n`;

          await native.writeFile(canonical, content);
          return {
            ok: true,
            message: "Project memory updated successfully.",
            appended: newRule,
          };
        } catch (e) {
          return { error: String(e) };
        }
      },
    }),
  } as const;
}
