import { generateText } from "ai";
import { gateway } from "@ai-sdk/gateway";
import { createOpenAI } from "@ai-sdk/openai";
import type { SearchAuth, SearchOptions, WebSearchResponse, WebSearchToolResult } from "./web-search";

const VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1";
const DEFAULT_VERCEL_SEARCH_MODEL = "openai/gpt-5.4";

type PerplexitySearchOptions = {
  searchDomainFilter?: string[];
};

function buildPerplexitySearchOptions(options?: SearchOptions): PerplexitySearchOptions | undefined {
  const allowed = options?.allowed_domains?.filter(Boolean) ?? [];
  const blocked = options?.blocked_domains?.filter(Boolean) ?? [];

  // AI Gateway does not allow mixing allowlist and denylist in searchDomainFilter.
  // When both are present, prefer explicit allowlist.
  if (allowed.length > 0) return { searchDomainFilter: allowed };
  if (blocked.length > 0) return { searchDomainFilter: blocked.map((domain) => `-${domain}`) };
  return undefined;
}

/**
 * Parse citations from Vercel AI SDK normalized `sources` output.
 */
export function extractVercelSources(sources: unknown): WebSearchToolResult[] {
  if (!Array.isArray(sources)) return [];

  const seen = new Set<string>();
  const citations: { title: string; url: string }[] = [];

  for (const source of sources) {
    if (!source || typeof source !== "object") continue;
    const entry = source as Record<string, unknown>;
    const url =
      typeof entry.url === "string"
        ? entry.url
        : typeof entry.href === "string"
          ? entry.href
          : null;

    if (!url || seen.has(url)) continue;
    seen.add(url);
    citations.push({
      title: typeof entry.title === "string" && entry.title.trim().length > 0 ? entry.title : url,
      url,
    });
  }

  if (citations.length === 0) return [];
  return [{ tool_use_id: "vercel-search", content: citations }];
}

function extractSourcesFromResult(result: unknown): WebSearchToolResult[] {
  const direct = extractVercelSources((result as any)?.sources);
  if (direct.length > 0) return direct;

  const stepSources = ((result as any)?.steps ?? [])
    .flatMap((step: any) => [step?.sources, ...(step?.toolResults ?? []).map((tool: any) => tool?.sources)]);

  return extractVercelSources(stepSources.flat());
}

function countVercelSearchCalls(results: WebSearchToolResult[]): number {
  return results.length > 0 ? 1 : 0;
}

/**
 * Perform a web search using Vercel AI Gateway via AI SDK.
 *
 * Uses AI Gateway built-in Perplexity Search tool so web search behavior is provider-agnostic.
 */
export async function vercelWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  const provider = createOpenAI({
    apiKey: auth.value,
    baseURL: VERCEL_AI_GATEWAY_BASE_URL,
  });

  const perplexitySearchOptions = buildPerplexitySearchOptions(options);

  const result = await generateText({
    model: provider(options?.model ?? DEFAULT_VERCEL_SEARCH_MODEL),
    prompt: query,
    ...(options?.system ? { system: options.system } : {}),
    tools: {
      perplexity_search: gateway.tools.perplexitySearch(perplexitySearchOptions),
    },
  });

  const normalizedSources = extractSourcesFromResult(result);

  return {
    query,
    results: normalizedSources,
    textResponse: (result as any)?.text ?? "",
    usage: {
      input_tokens: (result as any)?.usage?.inputTokens ?? 0,
      output_tokens: (result as any)?.usage?.outputTokens ?? 0,
      web_search_requests: countVercelSearchCalls(normalizedSources),
    },
  };
}
