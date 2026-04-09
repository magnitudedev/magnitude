import type { SearchAuth, SearchOptions } from "../../../src/tools/web-search";
import { runAnthropicDirectAdapter } from "./anthropic";
import { runCopilotDirectAdapter } from "./copilot";
import { runGeminiDirectAdapter } from "./gemini";
import { runOpenAIDirectAdapter } from "./openai";
import { runOpenRouterDirectAdapter } from "./openrouter";
import { runVercelDirectAdapter } from "./vercel";

type ProviderSlot = "openai" | "openrouter" | "vercel" | "github-copilot" | "google" | "anthropic";

export function runDirectAdapter(
  providerSlot: ProviderSlot,
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
) {
  switch (providerSlot) {
    case "openai":
      return runOpenAIDirectAdapter(query, auth, options);
    case "openrouter":
      return runOpenRouterDirectAdapter(query, auth, options);
    case "vercel":
      return runVercelDirectAdapter(query, auth, options);
    case "github-copilot":
      return runCopilotDirectAdapter(query, auth, options);
    case "google":
      return runGeminiDirectAdapter(query, auth, options);
    case "anthropic":
      return runAnthropicDirectAdapter(query, auth, options);
  }
}
