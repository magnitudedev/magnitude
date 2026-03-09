import OpenAI from "openai";
import type { SearchAuth, WebSearchResponse, WebSearchToolResult, SearchOptions } from "./web-search";

const CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses";

/**
 * Parse URL citations from an OpenAI Responses API output array.
 * Works for both SDK Response objects and raw Codex JSON.
 *
 * Tries two extraction paths:
 * 1. url_citation annotations on message output (has title + url)
 * 2. web_search_call action sources (url only, requires include param)
 */
function extractCitations(output: any[]): WebSearchToolResult[] {
  const results: WebSearchToolResult[] = [];

  // 1. Extract url_citation annotations from message output
  for (const item of output) {
    if (item.type === "message") {
      for (const content of item.content ?? []) {
        if (content.type === "output_text" && content.annotations) {
          const citations = content.annotations
            .filter((a: any) => a.type === "url_citation")
            .map((a: any) => ({ title: a.title ?? a.url, url: a.url }));
          if (citations.length > 0) {
            // Deduplicate by URL
            const seen = new Set<string>();
            const unique = citations.filter((c: { url: string }) => {
              if (seen.has(c.url)) return false;
              seen.add(c.url);
              return true;
            });
            results.push({ tool_use_id: "openai-search", content: unique });
          }
        }
      }
    }
  }

  // 2. Fallback: extract from web_search_call action sources
  if (results.length === 0) {
    const seen = new Set<string>();
    const fallback: { title: string; url: string }[] = [];
    for (const item of output) {
      if (item.type === "web_search_call" && item.action?.sources) {
        for (const source of item.action.sources) {
          const url = typeof source === "string" ? source : source.url;
          if (url && !seen.has(url)) {
            seen.add(url);
            fallback.push({ title: url, url });
          }
        }
      }
    }
    if (fallback.length > 0) {
      results.push({ tool_use_id: "openai-search", content: fallback });
    }
  }

  return results;
}

/**
 * Count actual web_search_call items in the output.
 */
function countSearchCalls(output: any[]): number {
  return output.filter((item: any) => item.type === "web_search_call").length;
}

/**
 * Perform a web search via the Codex endpoint (for OAuth/ChatGPT subscription users).
 *
 * The Codex endpoint requires:
 * - `input` as a message array (not a string)
 * - `stream: true` (always)
 * - `instructions` as a top-level field (always)
 *
 * Matches the request format in codex-stream.ts exactly.
 */
async function codexWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  const codexBody = {
    model: options?.model ?? "gpt-5.2",
    instructions: options?.system ?? "",
    input: [{ role: "user", content: query }],
    tools: [
      {
        type: "web_search",
        ...(options?.allowed_domains && options.allowed_domains.length > 0
          ? { filters: { allowed_domains: options.allowed_domains } }
          : {}),
      },
    ],
    include: ["web_search_call.action.sources"],
    stream: true,
    store: false,
  };

  const response = await fetch(CODEX_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${auth.value}`,
      ...(auth.accountId ? { "ChatGPT-Account-Id": auth.accountId } : {}),
    },
    body: JSON.stringify(codexBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI Codex web search error ${response.status}: ${errorText}`);
  }

  // Parse SSE stream and collect the completed response (same pattern as codex-stream.ts)
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completedResponse: any = null;
  let textResponse = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") break;

      try {
        const event = JSON.parse(data);
        if (event.type === "response.output_text.delta") {
          textResponse += event.delta ?? "";
        } else if (event.type === "response.completed") {
          completedResponse = event.response;
        }
      } catch {
        // Skip unparseable lines
      }
    }
  }

  // Extract citations from the completed response
  const outputItems = completedResponse?.output ?? [];
  const results = extractCitations(outputItems);

  return {
    query,
    results,
    textResponse: completedResponse?.output_text ?? textResponse,
    usage: {
      input_tokens: completedResponse?.usage?.input_tokens ?? 0,
      output_tokens: completedResponse?.usage?.output_tokens ?? 0,
      web_search_requests: countSearchCalls(outputItems),
    },
  };
}

/**
 * Perform a web search via the standard OpenAI Responses API (for API key users).
 */
async function apiKeyWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  const client = new OpenAI({ apiKey: auth.value });

  const response = await client.responses.create({
    model: options?.model ?? "gpt-5.2",
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

  const textResponse = response.output_text ?? "";
  const results = extractCitations(response.output);

  return {
    query,
    results,
    textResponse,
    usage: {
      input_tokens: response.usage?.input_tokens ?? 0,
      output_tokens: response.usage?.output_tokens ?? 0,
      web_search_requests: countSearchCalls(response.output),
    },
  };
}

/**
 * Perform a web search using OpenAI's Responses API with web_search tool.
 * Routes to Codex endpoint for OAuth users, standard API for key users.
 */
export async function openaiWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  if (auth.type === "oauth-token") {
    return codexWebSearch(query, auth, options);
  }
  return apiKeyWebSearch(query, auth, options);
}

// Quick test
if (import.meta.main) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    console.error("Set OPENAI_API_KEY to test OpenAI web search.");
    process.exit(1);
  }
  const result = await openaiWebSearch(
    "What is the current price of Bitcoin?",
    { type: "api-key", value: key },
  );
  console.log(JSON.stringify(result, null, 2));
}
