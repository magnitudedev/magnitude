import { safeJson } from "./capture-harness";
import type { SearchOptions } from "../../src/tools/web-search";

const VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1";
const FIXED_VERCEL_OPENAI_SEARCH_MODEL = "openai/gpt-5.4";

export interface VercelAiSdkCaptureDiagnostics {
  unsupportedToolWarning: boolean;
  requestToolsDropped: boolean;
  requestedWebSearchTool: boolean;
  hasCitations: boolean;
  warningTypes: string[];
}

export interface VercelAiSdkCaptureResult {
  request: unknown;
  response: unknown;
  responseRawText: string | null;
  streamEvents: unknown[];
  normalizedResult: unknown;
  diagnostics: VercelAiSdkCaptureDiagnostics;
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

function normalizeUsage(rawUsage: unknown, sourceCount: number, toolCallCount: number) {
  const usageObj = rawUsage && typeof rawUsage === "object" ? rawUsage as Record<string, unknown> : {};
  return {
    input_tokens: typeof usageObj.inputTokens === "number" ? usageObj.inputTokens : 0,
    output_tokens: typeof usageObj.outputTokens === "number" ? usageObj.outputTokens : 0,
    web_search_requests: toolCallCount > 0 ? toolCallCount : sourceCount > 0 ? 1 : 0,
  };
}

function parseWarningTypes(rawWarnings: unknown): string[] {
  if (!Array.isArray(rawWarnings)) return [];
  return rawWarnings
    .map((warning) => (warning && typeof warning === "object" ? (warning as Record<string, unknown>).type : undefined))
    .filter((type): type is string => typeof type === "string");
}

function hasUnsupportedToolWarning(rawWarnings: unknown): boolean {
  return parseWarningTypes(rawWarnings).includes("unsupported-tool");
}

function requestBodyHasEmptyTools(rawBody: unknown): boolean {
  if (typeof rawBody !== "string") return false;
  try {
    const parsed = JSON.parse(rawBody) as Record<string, unknown>;
    return Array.isArray(parsed.tools) && parsed.tools.length === 0;
  } catch {
    return false;
  }
}

function resolveOpenAIWebSearchTool(
  provider: Record<string, unknown>,
  options?: SearchOptions,
): { tool: unknown; helper: "webSearch" } {
  const toolsApi = (provider as any)?.tools;
  if (!toolsApi || typeof toolsApi !== "object" || typeof toolsApi.webSearch !== "function") {
    throw new Error("Vercel OpenAI webSearch helper not available in @ai-sdk/openai version");
  }

  const config = options?.allowed_domains && options.allowed_domains.length > 0
    ? { filters: { allowed_domains: options.allowed_domains } }
    : {};

  return { tool: toolsApi.webSearch(config), helper: "webSearch" };
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
  }) as Record<string, unknown>;

  const { tool: openAIWebSearchTool, helper } = resolveOpenAIWebSearchTool(provider, options);

  const requestArgs = {
    model: FIXED_VERCEL_OPENAI_SEARCH_MODEL,
    prompt: query,
    ...(options?.system ? { system: options.system } : {}),
    tools: {
      web_search: openAIWebSearchTool,
    },
    helper,
  };

  const result = await generateText({
    model: (provider as any).responses(FIXED_VERCEL_OPENAI_SEARCH_MODEL),
    prompt: query,
    ...(options?.system ? { system: options.system } : {}),
    tools: {
      web_search: openAIWebSearchTool as any,
    },
  });

  const citations = normalizeSources((result as any)?.sources);
  const warningTypes = parseWarningTypes((result as any)?.warnings);
  const requestedWebSearchTool = Boolean((requestArgs as any)?.tools?.web_search);
  const requestToolsDropped = requestBodyHasEmptyTools((result as any)?.request?.body);
  const toolCallCount = Array.isArray((result as any)?.toolCalls) ? (result as any).toolCalls.length : 0;
  const diagnostics: VercelAiSdkCaptureDiagnostics = {
    unsupportedToolWarning: hasUnsupportedToolWarning((result as any)?.warnings),
    requestToolsDropped,
    requestedWebSearchTool,
    hasCitations: citations.length > 0,
    warningTypes,
  };

  const normalizedResult = {
    query,
    results: citations.length > 0 ? [{ tool_use_id: "vercel-search", content: citations }] : [],
    textResponse: (result as any)?.text ?? "",
    usage: normalizeUsage((result as any)?.usage, citations.length, toolCallCount),
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
    diagnostics,
  };
}

export const __testOnly = {
  normalizeSources,
  parseWarningTypes,
  hasUnsupportedToolWarning,
  requestBodyHasEmptyTools,
  resolveOpenAIWebSearchTool,
};