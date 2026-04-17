import type { SearchAuth, SearchOptions } from "../../../src/tools/web-search";
import { runAnthropicDirectAdapter } from "./anthropic";
import { runOpenAIDirectAdapter } from "./openai";
import { runOpenRouterDirectAdapter } from "./openrouter";
import { runVercelDirectAdapter } from "./vercel";

type ProviderSlot = "openai" | "openrouter" | "vercel" | "anthropic";

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
    case "anthropic":
      return runAnthropicDirectAdapter(query, auth, options);
  }
}
