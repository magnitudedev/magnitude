import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import type { SearchAuth, SearchOptions, WebSearchResponse, WebSearchToolResult } from "./web-search";

const VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1";
const DEFAULT_VERCEL_SEARCH_MODEL = "openai/gpt-5.4";

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

function resolveOpenAIWebSearchTool(
  provider: ReturnType<typeof createOpenAI>,
  options?: SearchOptions,
): unknown {
  const toolsApi = (provider as any)?.tools;
  if (!toolsApi || typeof toolsApi !== "object" || typeof toolsApi.webSearch !== "function") {
    throw new Error("Vercel OpenAI webSearch helper not available in @ai-sdk/openai version");
  }

  const config = options?.allowed_domains && options.allowed_domains.length > 0
    ? { filters: { allowed_domains: options.allowed_domains } }
    : {};

  return toolsApi.webSearch(config);
}

function extractSourcesFromSteps(result: unknown): WebSearchToolResult[] {
  const steps = (result as any)?.steps;
  if (!Array.isArray(steps)) return [];

  const aggregated: unknown[] = [];
  for (const step of steps) {
    if (!step || typeof step !== "object") continue;

    if (Array.isArray((step as any).sources)) {
      aggregated.push(...(step as any).sources);
    }

    const toolResults = (step as any).toolResults;
    if (!Array.isArray(toolResults)) continue;
    for (const toolResult of toolResults) {
      if (Array.isArray((toolResult as any)?.sources)) {
        aggregated.push(...(toolResult as any).sources);
      }
    }
  }

  return extractVercelSources(aggregated);
}

function extractSourcesFromText(text: string): WebSearchToolResult[] {
  if (!text) return [];
  const seen = new Set<string>();
  const citations: { title: string; url: string }[] = [];
  const markdownLinkRegex = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g;
  let match: RegExpExecArray | null = markdownLinkRegex.exec(text);
  while (match) {
    const title = match[1]?.trim();
    const url = match[2];
    if (url && !seen.has(url)) {
      seen.add(url);
      citations.push({ title: title && title.length > 0 ? title : url, url });
    }
    match = markdownLinkRegex.exec(text);
  }
  return citations.length > 0 ? [{ tool_use_id: "vercel-search", content: citations }] : [];
}

function mergeSourceBuckets(...buckets: WebSearchToolResult[][]): WebSearchToolResult[] {
  const seen = new Set<string>();
  const merged: { title: string; url: string }[] = [];

  for (const bucket of buckets) {
    for (const result of bucket) {
      if (!result || !Array.isArray(result.content)) continue;
      for (const citation of result.content) {
        if (!citation || typeof citation.url !== "string" || citation.url.length === 0) continue;
        if (seen.has(citation.url)) continue;
        seen.add(citation.url);
        merged.push({
          title: typeof citation.title === "string" && citation.title.length > 0 ? citation.title : citation.url,
          url: citation.url,
        });
      }
    }
  }

  return merged.length > 0 ? [{ tool_use_id: "vercel-search", content: merged }] : [];
}

/**
 * Perform a web search via OpenAI through Vercel AI Gateway using AI SDK.
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

  const webSearchTool = resolveOpenAIWebSearchTool(provider, options);

  const result = await generateText({
    model: provider.responses(options?.model ?? DEFAULT_VERCEL_SEARCH_MODEL),
    prompt: query,
    ...(options?.system ? { system: options.system } : {}),
    tools: {
      web_search: webSearchTool as any,
    },
  });

  const textResponse = (result as any)?.text ?? "";
  const normalizedFromSources = extractVercelSources((result as any)?.sources);
  const normalizedFromSteps = extractSourcesFromSteps(result);
  const normalizedFromText = extractSourcesFromText(textResponse);
  const normalizedSources = mergeSourceBuckets(
    normalizedFromSources,
    normalizedFromSteps,
    normalizedFromText,
  );

  const webSearchRequests =
    Array.isArray((result as any)?.toolCalls) && (result as any).toolCalls.length > 0
      ? (result as any).toolCalls.length
      : normalizedSources.length > 0
        ? 1
        : 0;

  return {
    query,
    results: normalizedSources,
    textResponse,
    usage: {
      input_tokens: (result as any)?.usage?.inputTokens ?? 0,
      output_tokens: (result as any)?.usage?.outputTokens ?? 0,
      web_search_requests: webSearchRequests,
    },
  };
}
