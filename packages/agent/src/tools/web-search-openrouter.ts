import type { SearchAuth, SearchOptions, WebSearchResponse, WebSearchToolResult } from "./web-search";

const OPENROUTER_RESPONSES_ENDPOINT = "https://openrouter.ai/api/v1/responses";
const FIXED_OPENROUTER_SEARCH_MODEL = "openai/gpt-5.4";

type OpenRouterAnnotation = {
  type?: string;
  url?: string;
  title?: string;
};

type OpenRouterOutputItem = {
  type?: string;
  content?: Array<{
    type?: string;
    text?: string;
    annotations?: OpenRouterAnnotation[];
  }>;
  annotations?: OpenRouterAnnotation[];
};

type OpenRouterResponse = {
  output?: OpenRouterOutputItem[];
  output_text?: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    server_tool_use?: {
      web_search_requests?: number;
    };
  };
};

function extractText(output: OpenRouterOutputItem[], fallback = ""): string {
  const parts: string[] = [];

  for (const item of output) {
    if (!Array.isArray(item.content)) continue;
    for (const content of item.content) {
      if (content.type === "output_text" && typeof content.text === "string") {
        parts.push(content.text);
      }
    }
  }

  return parts.join("").trim() || fallback;
}

function extractCitations(output: OpenRouterOutputItem[]): WebSearchToolResult[] {
  const seen = new Set<string>();
  const citations: { title: string; url: string }[] = [];

  const visitAnnotations = (annotations?: OpenRouterAnnotation[]) => {
    for (const annotation of annotations ?? []) {
      if (annotation.type !== "url_citation" || !annotation.url) continue;
      if (seen.has(annotation.url)) continue;
      seen.add(annotation.url);
      citations.push({
        title: annotation.title ?? annotation.url,
        url: annotation.url,
      });
    }
  };

  for (const item of output) {
    visitAnnotations(item.annotations);
    for (const content of item.content ?? []) {
      visitAnnotations(content.annotations);
    }
  }

  if (citations.length === 0) return [];
  return [{ tool_use_id: "openrouter-search", content: citations }];
}

function countSearchRequests(response: OpenRouterResponse, results: WebSearchToolResult[]): number {
  const explicit = response.usage?.server_tool_use?.web_search_requests;
  if (typeof explicit === "number") return explicit;
  return results.length > 0 ? 1 : 0;
}

function buildOpenRouterWebSearchParameters(options?: SearchOptions): Record<string, unknown> | undefined {
  const parameters = {
    ...(options?.allowed_domains && options.allowed_domains.length > 0
      ? { allowed_domains: options.allowed_domains }
      : {}),
    ...(options?.blocked_domains && options.blocked_domains.length > 0
      ? { excluded_domains: options.blocked_domains }
      : {}),
  };

  return Object.keys(parameters).length > 0 ? parameters : undefined;
}

export async function openrouterWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  const parameters = buildOpenRouterWebSearchParameters(options);
  const body = {
    model: FIXED_OPENROUTER_SEARCH_MODEL,
    input: query,
    tools: [
      {
        type: "openrouter:web_search",
        ...(parameters ? { parameters } : {}),
      },
    ],
    ...(options?.system ? { instructions: options.system } : {}),
  };

  const response = await fetch(OPENROUTER_RESPONSES_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${auth.value}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenRouter web search error ${response.status}: ${errorText}`);
  }

  const json = await response.json() as OpenRouterResponse;
  const output = Array.isArray(json.output) ? json.output : [];
  const results = extractCitations(output);
  const textResponse = extractText(output, json.output_text ?? "");

  return {
    query,
    results,
    textResponse,
    usage: {
      input_tokens: json.usage?.input_tokens ?? 0,
      output_tokens: json.usage?.output_tokens ?? 0,
      web_search_requests: countSearchRequests(json, results),
    },
  };
}

export const __testOnly = {
  extractCitations,
  extractText,
  countSearchRequests,
  buildOpenRouterWebSearchParameters,
};

// Quick test
if (import.meta.main) {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) {
    console.error("Set OPENROUTER_API_KEY to test OpenRouter web search.");
    process.exit(1);
  }
  const result = await openrouterWebSearch(
    "What is the current price of Bitcoin?",
    { type: "api-key", value: key },
  );
  console.log(JSON.stringify(result, null, 2));
}
