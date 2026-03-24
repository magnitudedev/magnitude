/**
 * Web Search Router
 *
 * Routes web search requests to the appropriate provider based on the user's
 * primary model selection. Exports shared types used by all provider implementations.
 */

import { Effect } from 'effect'
import { ProviderAuth, ProviderState } from '@magnitudedev/providers'
import type { AuthInfo } from '@magnitudedev/providers'

// =============================================================================
// Shared Types
// =============================================================================

export interface WebSearchResult {
  title: string;
  url: string;
}

export interface WebSearchToolResult {
  tool_use_id: string;
  content: WebSearchResult[];
}

export interface WebSearchResponse {
  query: string;
  results: (WebSearchToolResult | string)[];
  textResponse: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
    web_search_requests: number;
  };
}

export interface SearchAuth {
  type: "oauth-token" | "api-key";
  value: string;
  accountId?: string;  // ChatGPT account ID for Codex endpoint
}

export interface SearchOptions {
  system?: string;
  allowed_domains?: string[];
  blocked_domains?: string[];
  model?: string;
  max_tokens?: number;
}

// =============================================================================
// Provider Detection
// =============================================================================

type SearchProvider = "anthropic" | "openai" | "gemini";

function resolveAnthropicAuth(auth?: AuthInfo): SearchAuth {
  // 1. OAuth (always first)
  if (auth?.type === "oauth") return { type: "oauth-token", value: auth.accessToken };
  // 2. Stored API key
  if (auth?.type === "api") return { type: "api-key", value: auth.key };
  // 3. Environment variable
  const envKey = process.env.ANTHROPIC_API_KEY;
  if (envKey) return { type: "api-key", value: envKey };
  throw new Error("No Anthropic auth available for web search. Set ANTHROPIC_API_KEY or authenticate via the app.");
}

function resolveOpenAIAuth(auth?: AuthInfo): SearchAuth {
  // 1. OAuth (always first)
  if (auth?.type === "oauth") return { type: "oauth-token", value: auth.accessToken, accountId: auth.accountId };
  // 2. Stored API key
  if (auth?.type === "api") return { type: "api-key", value: auth.key };
  // 3. Environment variable
  const envKey = process.env.OPENAI_API_KEY;
  if (envKey) return { type: "api-key", value: envKey };
  throw new Error("No OpenAI auth available for web search. Set OPENAI_API_KEY or authenticate via the app.");
}

function resolveGeminiAuth(auth?: AuthInfo): SearchAuth {
  // 1. Stored API key (Gemini doesn't have OAuth)
  if (auth?.type === "api") return { type: "api-key", value: auth.key };
  // 2. Environment variable
  const envKey = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY;
  if (envKey) return { type: "api-key", value: envKey };
  throw new Error("No Google API key available for web search. Set GOOGLE_API_KEY or GEMINI_API_KEY.");
}

/**
 * Resolve auth for the MAGNITUDE_SEARCH_PROVIDER override.
 * Probes stored auth and env vars for the specified search provider.
 */
function detectSearchProvider(providerId: string | null): SearchProvider {
  const override = process.env.MAGNITUDE_SEARCH_PROVIDER as SearchProvider | undefined;
  if (override) {
    if (override !== "anthropic" && override !== "openai" && override !== "gemini") {
      throw new Error(
        `Invalid MAGNITUDE_SEARCH_PROVIDER value "${override}". Must be one of: anthropic, openai, gemini.`
      );
    }
    return override;
  }

  switch (providerId) {
    case "anthropic":
      return "anthropic";
    case "openai":
      return "openai";
    case "google":
    case "google-vertex":
      return "gemini";
    case "google-vertex-anthropic":
    case "amazon-bedrock":
    case null:
      return "anthropic";
    default:
      throw new Error(
        `Web search is not supported with the "${providerId}" provider. ` +
        `To enable web search, set MAGNITUDE_SEARCH_PROVIDER to "anthropic", "openai", or "gemini".`
      );
  }
}

function resolveSearchAuth(provider: SearchProvider): Effect.Effect<SearchAuth, Error, ProviderAuth> {
  return Effect.gen(function* () {
    const auth = yield* ProviderAuth

    switch (provider) {
      case "anthropic":
        return resolveAnthropicAuth(yield* auth.getAuth("anthropic"));
      case "openai":
        return resolveOpenAIAuth(yield* auth.getAuth("openai"));
      case "gemini":
        return resolveGeminiAuth(yield* auth.getAuth("google"));
    }
  })
}

// =============================================================================
// Router
// =============================================================================

/**
 * Perform a web search using the provider determined by the user's primary model.
 */
export function webSearch(
  query: string,
  options?: SearchOptions,
): Effect.Effect<WebSearchResponse, Error, ProviderState | ProviderAuth> {
  return Effect.gen(function* () {
    const providerState = yield* ProviderState
    const current = yield* providerState.peek('orchestrator')
    const provider = detectSearchProvider(current?.model.providerId ?? null)
    const auth = yield* resolveSearchAuth(provider)

    switch (provider) {
      case "anthropic": {
        const { anthropicWebSearch } = yield* Effect.promise(() => import("./web-search-anthropic"));
        return yield* Effect.promise(() => anthropicWebSearch(query, auth, options));
      }
      case "openai": {
        const { openaiWebSearch } = yield* Effect.promise(() => import("./web-search-openai"));
        return yield* Effect.promise(() => openaiWebSearch(query, auth, options));
      }
      case "gemini": {
        const { geminiWebSearch } = yield* Effect.promise(() => import("./web-search-gemini"));
        return yield* Effect.promise(() => geminiWebSearch(query, auth, options));
      }
    }
  })
}

/**
 * Streaming web search (Anthropic-only).
 * Caller must provide already-resolved auth.
 */
export async function* webSearchStream(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): AsyncGenerator<
  | { type: "search_started"; query: string }
  | { type: "search_result"; result: WebSearchToolResult }
  | { type: "text_delta"; text: string }
  | { type: "done"; response: WebSearchResponse }
> {
  const { anthropicWebSearchStream } = await import("./web-search-anthropic");
  yield* anthropicWebSearchStream(query, auth, options);
}
