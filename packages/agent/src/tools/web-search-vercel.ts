import OpenAI from "openai";
import type { SearchAuth, SearchOptions, WebSearchResponse, WebSearchToolResult } from "./web-search";

const VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1";
const FIXED_VERCEL_OPENAI_SEARCH_MODEL = "openai/gpt-5.4";

/**
 * Parse citations from Vercel AI Gateway's OpenAI-native Responses API output array.
 *
 * This backend intentionally targets Vercel's provider-native OpenAI web-search path
 * with a fixed GPT-5.4 model. Canonical sources come from url_citation annotations,
 * with web_search_call.action.sources used only as a fallback when annotations are absent.
 */
export function extractVercelCitations(output: any[]): WebSearchToolResult[] {
  const results: WebSearchToolResult[] = [];

  for (const item of output) {
    if (item.type !== "message") continue;

    for (const content of item.content ?? []) {
      if (content.type !== "output_text" || !Array.isArray(content.annotations)) continue;

      const seen = new Set<string>();
      const citations = content.annotations
        .filter((annotation: any) => annotation?.type === "url_citation" && typeof annotation.url === "string")
        .map((annotation: any) => ({
          title: annotation.title ?? annotation.url,
          url: annotation.url,
        }))
        .filter((citation: { url: string }) => {
          if (seen.has(citation.url)) return false;
          seen.add(citation.url);
          return true;
        });

      if (citations.length > 0) {
        results.push({ tool_use_id: "vercel-search", content: citations });
      }
    }
  }

  if (results.length > 0) return results;

  const seen = new Set<string>();
  const fallback: { title: string; url: string }[] = [];

  for (const item of output) {
    if (item?.type !== "web_search_call" || !Array.isArray(item?.action?.sources)) continue;

    for (const source of item.action.sources) {
      const url = typeof source === "string" ? source : source?.url;
      const title = typeof source === "string" ? source : source?.title ?? source?.url;
      if (!url || seen.has(url)) continue;
      seen.add(url);
      fallback.push({ title, url });
    }
  }

  if (fallback.length > 0) {
    results.push({ tool_use_id: "vercel-search", content: fallback });
  }

  return results;
}

export function countVercelSearchCalls(output: any[]): number {
  return output.filter((item: any) => item.type === "web_search_call").length;
}

/**
 * Perform a web search using Vercel AI Gateway's provider-native OpenAI Responses path.
 *
 * Vercel search intentionally always uses the fixed GPT-5.4 search model and ignores
 * caller model overrides so this backend stays on the known-good OpenAI-native search path.
 */
export async function vercelWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  const client = new OpenAI({
    apiKey: auth.value,
    baseURL: VERCEL_AI_GATEWAY_BASE_URL,
  });

  const response = await client.responses.create({
    model: FIXED_VERCEL_OPENAI_SEARCH_MODEL,
    input: query,
    tools: [
      {
        type: "web_search",
        ...(options?.allowed_domains && options.allowed_domains.length > 0
          ? { filters: { allowed_domains: options.allowed_domains } }
          : {}),
      },
    ],
    include: ["web_search_call.action.sources"],
    ...(options?.system ? { instructions: options.system } : {}),
  });

  return {
    query,
    results: extractVercelCitations(response.output),
    textResponse: response.output_text ?? "",
    usage: {
      input_tokens: response.usage?.input_tokens ?? 0,
      output_tokens: response.usage?.output_tokens ?? 0,
      web_search_requests: countVercelSearchCalls(response.output),
    },
  };
}
