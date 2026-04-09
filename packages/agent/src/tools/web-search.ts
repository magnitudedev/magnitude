/**
 * Web Search Router
 *
 * Routes web search requests to the appropriate backend based on the user's
 * current provider selection. Exports shared types used by all provider implementations.
 */

import { Effect } from 'effect'
import { ProviderAuth, ProviderState } from '@magnitudedev/providers'
import type { AuthInfo } from '@magnitudedev/providers'
import { MAGNITUDE_SLOTS } from '../model-slots'

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

type SearchProvider = "anthropic" | "openai" | "gemini" | "openrouter" | "vercel" | "github-copilot";

const SEARCH_PROVIDER_OVERRIDES = ["anthropic", "openai", "gemini", "openrouter", "vercel", "github-copilot"] as const;
const SEARCHABLE_PROVIDER_SLOTS = ['lead', ...MAGNITUDE_SLOTS.filter((slot) => slot !== 'lead')] as const;

function resolveAnthropicAuth(auth?: AuthInfo): SearchAuth {
  if (auth?.type === "oauth") return { type: "oauth-token", value: auth.accessToken };
  if (auth?.type === "api") return { type: "api-key", value: auth.key };
  const envKey = process.env.ANTHROPIC_API_KEY;
  if (envKey) return { type: "api-key", value: envKey };
  throw new Error("No Anthropic auth available for web search. Set ANTHROPIC_API_KEY or authenticate via the app.");
}

function resolveOpenAIAuth(auth?: AuthInfo): SearchAuth {
  if (auth?.type === "oauth") return { type: "oauth-token", value: auth.accessToken, accountId: auth.accountId };
  if (auth?.type === "api") return { type: "api-key", value: auth.key };
  const envKey = process.env.OPENAI_API_KEY;
  if (envKey) return { type: "api-key", value: envKey };
  throw new Error("No OpenAI auth available for web search. Set OPENAI_API_KEY or authenticate via the app.");
}

function resolveGeminiAuth(auth?: AuthInfo): SearchAuth {
  if (auth?.type === "api") return { type: "api-key", value: auth.key };
  const envKey = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY;
  if (envKey) return { type: "api-key", value: envKey };
  throw new Error("No Google API key available for web search. Set GOOGLE_API_KEY or GEMINI_API_KEY.");
}

function resolveOpenRouterAuth(auth?: AuthInfo): SearchAuth {
  if (auth?.type === "api") return { type: "api-key", value: auth.key };
  const envKey = process.env.OPENROUTER_API_KEY;
  if (envKey) return { type: "api-key", value: envKey };
  throw new Error("No OpenRouter auth available for web search. Set OPENROUTER_API_KEY or authenticate via the app.");
}

export function resolveVercelAuth(auth?: AuthInfo): SearchAuth {
  if (auth?.type === "api") return { type: "api-key", value: auth.key };
  const envKey = process.env.AI_GATEWAY_API_KEY ?? process.env.VERCEL_API_KEY;
  if (envKey) return { type: "api-key", value: envKey };
  throw new Error("No Vercel AI Gateway API key available for web search. Set AI_GATEWAY_API_KEY or authenticate Vercel in the app.");
}

export function resolveCopilotAuth(auth?: AuthInfo): SearchAuth {
  if (auth?.type === "oauth") return { type: "oauth-token", value: auth.accessToken };
  throw new Error("No GitHub Copilot OAuth session available for web search. Authenticate GitHub Copilot in the app.");
}

export function detectSearchProvider(providerId: string | null): SearchProvider {
  const override = process.env.MAGNITUDE_SEARCH_PROVIDER as SearchProvider | undefined;
  if (override) {
    if (!SEARCH_PROVIDER_OVERRIDES.includes(override)) {
      throw new Error(
        `Invalid MAGNITUDE_SEARCH_PROVIDER value "${override}". Must be one of: ${SEARCH_PROVIDER_OVERRIDES.join(', ')}.`
      );
    }
    return override;
  }

  switch (providerId) {
    case "anthropic":
      return "anthropic";
    case "openai":
      return "openai";
    case "openrouter":
      return "openrouter";
    case "google":
      return "gemini";
    case "vercel":
      return "vercel";
    case "github-copilot":
      return "github-copilot";
    default:
      throw new Error(
        `Web search is not supported with the "${providerId}" provider. ` +
        `To enable web search, set MAGNITUDE_SEARCH_PROVIDER to ${SEARCH_PROVIDER_OVERRIDES.join(', ')}.`
      );
  }
}

function resolveSearchAuth(provider: SearchProvider): Effect.Effect<SearchAuth, Error, ProviderAuth> {
  return Effect.gen(function* () {
    const auth = yield* ProviderAuth;

    switch (provider) {
      case "anthropic":
        return resolveAnthropicAuth(yield* auth.getAuth("anthropic"));
      case "openai":
        return resolveOpenAIAuth(yield* auth.getAuth("openai"));
      case "gemini":
        return resolveGeminiAuth(yield* auth.getAuth("google"));
      case "openrouter":
        return resolveOpenRouterAuth(yield* auth.getAuth("openrouter"));
      case "vercel":
        return resolveVercelAuth(yield* auth.getAuth("vercel"));
      case "github-copilot":
        return resolveCopilotAuth(yield* auth.getAuth("github-copilot"));
    }
  });
}

function tryDetectSearchProvider(providerId: string | null): SearchProvider | null {
  try {
    return detectSearchProvider(providerId);
  } catch {
    return null;
  }
}

function getUnsupportedSearchError(): Error {
  return new Error(
    `No supported web-search backend is configured on the lead or worker slots. ` +
    `To enable web search, set MAGNITUDE_SEARCH_PROVIDER to ${SEARCH_PROVIDER_OVERRIDES.join(', ')}.`
  );
}

function selectSearchBackend(): Effect.Effect<
  { provider: SearchProvider; modelId?: string },
  Error,
  ProviderState
> {
  return Effect.gen(function* () {
    const providerState = yield* ProviderState;

    for (const slot of SEARCHABLE_PROVIDER_SLOTS) {
      const current = yield* providerState.peek(slot);
      const provider = tryDetectSearchProvider(current?.model.providerId ?? null);
      if (provider) {
        return {
          provider,
          modelId: current?.model.id,
        };
      }
    }

    return yield* Effect.fail(getUnsupportedSearchError());
  });
}

// =============================================================================
// Router
// =============================================================================

/**
 * Perform a web search using the lead provider first, then worker providers.
 * Arbitrary connected providers are intentionally ignored.
 */
export function webSearch(
  query: string,
  options?: SearchOptions,
): Effect.Effect<WebSearchResponse, Error, ProviderState | ProviderAuth> {
  return Effect.gen(function* () {
    const selection = yield* selectSearchBackend();
    const auth = yield* resolveSearchAuth(selection.provider);

    switch (selection.provider) {
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
      case "openrouter": {
        const { openrouterWebSearch } = yield* Effect.promise(() => import("./web-search-openrouter"));
        return yield* Effect.promise(() => openrouterWebSearch(query, auth, options));
      }
      case "vercel": {
        const { vercelWebSearch } = yield* Effect.promise(() => import("./web-search-vercel"));
        return yield* Effect.promise(() => vercelWebSearch(query, auth, options));
      }
      case "github-copilot": {
        const { copilotWebSearch } = yield* Effect.promise(() => import("./web-search-copilot"));
        return yield* Effect.promise(() => copilotWebSearch(query, auth, options));
      }
    }
  });
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
