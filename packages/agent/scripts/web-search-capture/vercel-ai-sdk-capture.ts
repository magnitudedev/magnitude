import { safeJson } from "./capture-harness";
import type { SearchOptions } from "../../src/tools/web-search";

const VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1";
const FIXED_VERCEL_OPENAI_SEARCH_MODEL = "openai/gpt-5.4";

export interface VercelAiSdkCaptureResult {
  request: unknown;
  response: unknown;
  responseRawText: string | null;
  streamEvents: unknown[];
  normalizedResult: unknown;
}

type Citation = { title: string; url: string };

function normalizeSources(rawSources: unknown): Citation[] {
  if (!Array.isArray(rawSources)) return [];
  const seen = new Set<string>();
  const citations: Citation[] = [];

  for (const source of rawSources) {
    if (!source || typeof source !== "object") continue;
    const obj = source as Record<string, unknown>;
    const url = typeof obj.url === "string"
      ? obj.url
      : typeof obj.href === "string"
        ? obj.href
        : null;
    if (!url || seen.has(url)) continue;
    seen.add(url);
    citations.push({
      title: typeof obj.title === "string" && obj.title.trim().length > 0 ? obj.title : url,
      url,
    });
  }

  return citations;
}

function normalizeUsage(rawUsage: unknown, sourceCount: number) {
  const usageObj = rawUsage && typeof rawUsage === "object" ? rawUsage as Record<string, unknown> : {};
  return {
    input_tokens: typeof usageObj.inputTokens === "number" ? usageObj.inputTokens : 0,
    output_tokens: typeof usageObj.outputTokens === "number" ? usageObj.outputTokens : 0,
    web_search_requests: sourceCount > 0 ? 1 : 0,
  };
}

export async function runVercelAiSdkCapture(
  query: string,
  apiKey: string,
  options?: SearchOptions,
): Promise<VercelAiSdkCaptureResult> {
  const [{ generateText }, { createOpenAI }] = await Promise.all([
    import("ai"),
    import("@ai-sdk/openai"),
  ]);

  const provider = createOpenAI({
    apiKey,
    baseURL: VERCEL_AI_GATEWAY_BASE_URL,
  });

  const webSearchToolFactory =
    (provider as any)?.tools?.webSearchPreview ??
    (provider as any)?.tools?.webSearch;

  if (typeof webSearchToolFactory !== "function") {
    throw new TypeError("OpenAI AI SDK web search tool is unavailable on this @ai-sdk/openai version");
  }

  const requestArgs = {
    model: FIXED_VERCEL_OPENAI_SEARCH_MODEL,
    prompt: query,
    ...(options?.system ? { system: options.system } : {}),
    tools: {
      web_search: webSearchToolFactory({}),
    },
  };

  const result = await generateText({
    model: provider(FIXED_VERCEL_OPENAI_SEARCH_MODEL),
    prompt: query,
    ...(options?.system ? { system: options.system } : {}),
    tools: {
      web_search: webSearchToolFactory({}),
    },
  });

  const citations = normalizeSources((result as any)?.sources);
  const normalizedResult = {
    query,
    results: citations.length > 0 ? [{ tool_use_id: "vercel-search", content: citations }] : [],
    textResponse: (result as any)?.text ?? "",
    usage: normalizeUsage((result as any)?.usage, citations.length),
  };

  return {
    request: safeJson({
      present: true,
      client: { baseURL: VERCEL_AI_GATEWAY_BASE_URL, sdk: "ai" },
      args: requestArgs,
    }),
    response: safeJson({
      present: true,
      value: result,
    }),
    responseRawText: null,
    streamEvents: [],
    normalizedResult,
  };
}

export const __testOnly = {
  normalizeSources,
};