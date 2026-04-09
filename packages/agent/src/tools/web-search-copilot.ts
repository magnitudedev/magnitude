import type { SearchAuth, SearchOptions, WebSearchResponse, WebSearchToolResult } from "./web-search";

const COPILOT_RESPONSES_ENDPOINT = "https://api.githubcopilot.com/responses";
const DEFAULT_COPILOT_SEARCH_MODEL = "gpt-5.4";
const COPILOT_TIMEOUT_MS = 180_000;

type CopilotCitation = { title: string; url: string };

type CopilotOutputTextContent = {
  type?: string;
  text?: string;
  annotations?: Array<{
    type?: string;
    url?: string;
    title?: string;
  }>;
};

type CopilotOutputItem = {
  type?: string;
  content?: CopilotOutputTextContent[];
  action?: {
    sources?: Array<string | { url?: string; href?: string; title?: string }>;
  };
};

type CopilotResponse = {
  output?: CopilotOutputItem[];
  output_text?: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    server_tool_use?: {
      web_search_requests?: number;
    };
    web_search_requests?: number;
  };
};

function buildCopilotHeaders(auth: SearchAuth): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${auth.value}`,
    "Openai-Intent": "conversation-edits",
    "x-initiator": "user",
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
  };
}

function buildCopilotWebSearchTool(options?: SearchOptions): Record<string, unknown> {
  const tool: Record<string, unknown> = { type: "web_search" };

  if (options?.allowed_domains && options.allowed_domains.length > 0) {
    tool.filters = { allowed_domains: options.allowed_domains };
  }

  return tool;
}

export function buildCopilotWebSearchRequest(
  query: string,
  options?: SearchOptions,
): Record<string, unknown> {
  return {
    model: options?.model ?? DEFAULT_COPILOT_SEARCH_MODEL,
    input: query,
    stream: false,
    tools: [buildCopilotWebSearchTool(options)],
    include: ["web_search_call.action.sources"],
    reasoning: { effort: "none" },
    ...(options?.system ? { instructions: options.system } : {}),
  };
}

function normalizeSourceUrl(source: unknown): string | null {
  if (typeof source === "string") return source;
  if (!source || typeof source !== "object") return null;
  const entry = source as Record<string, unknown>;
  if (typeof entry.url === "string" && entry.url.length > 0) return entry.url;
  if (typeof entry.href === "string" && entry.href.length > 0) return entry.href;
  return null;
}

function normalizeSourceTitle(source: unknown, fallbackUrl: string): string {
  if (!source || typeof source !== "object") return fallbackUrl;
  const entry = source as Record<string, unknown>;
  return typeof entry.title === "string" && entry.title.length > 0 ? entry.title : fallbackUrl;
}

export function extractCopilotCitations(output: CopilotOutputItem[]): WebSearchToolResult[] {
  const merged: CopilotCitation[] = [];
  const seen = new Set<string>();

  for (const item of output) {
    if (item.type !== "message") continue;
    for (const content of item.content ?? []) {
      if (content.type !== "output_text" || !Array.isArray(content.annotations)) continue;
      for (const annotation of content.annotations) {
        if (annotation?.type !== "url_citation" || typeof annotation.url !== "string" || annotation.url.length === 0) {
          continue;
        }
        if (seen.has(annotation.url)) continue;
        seen.add(annotation.url);
        merged.push({
          title: annotation.title ?? annotation.url,
          url: annotation.url,
        });
      }
    }
  }

  for (const item of output) {
    if (item.type !== "web_search_call" || !Array.isArray(item.action?.sources)) continue;
    for (const source of item.action!.sources!) {
      const url = normalizeSourceUrl(source);
      if (!url || seen.has(url)) continue;
      seen.add(url);
      merged.push({
        title: normalizeSourceTitle(source, url),
        url,
      });
    }
  }

  if (merged.length === 0) return [];
  return [{ tool_use_id: "copilot-search", content: merged }];
}

export function extractCopilotTextResponse(response: CopilotResponse): string {
  if (typeof response.output_text === "string" && response.output_text.length > 0) {
    return response.output_text;
  }

  const parts: string[] = [];
  for (const item of response.output ?? []) {
    if (item.type !== "message") continue;
    for (const content of item.content ?? []) {
      if (content.type === "output_text" && typeof content.text === "string" && content.text.length > 0) {
        parts.push(content.text);
      }
    }
  }

  return parts.join("\n");
}

export function countCopilotSearchCalls(output: CopilotOutputItem[]): number {
  return output.filter((item) => item.type === "web_search_call").length;
}

function resolveSearchRequestCount(response: CopilotResponse, output: CopilotOutputItem[]): number {
  const counted = countCopilotSearchCalls(output);
  if (counted > 0) return counted;

  const usageSearchRequests =
    response.usage?.server_tool_use?.web_search_requests ?? response.usage?.web_search_requests;

  return typeof usageSearchRequests === "number" ? usageSearchRequests : 0;
}

async function parseErrorBody(response: Response): Promise<string> {
  try {
    const text = await response.text();
    return text || "<empty body>";
  } catch {
    return "<unable to read body>";
  }
}

export async function copilotWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), COPILOT_TIMEOUT_MS);

  try {
    const response = await fetch(COPILOT_RESPONSES_ENDPOINT, {
      method: "POST",
      headers: buildCopilotHeaders(auth),
      body: JSON.stringify(buildCopilotWebSearchRequest(query, options)),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await parseErrorBody(response);
      throw new Error(`GitHub Copilot web search error ${response.status}: ${errorText}`);
    }

    const json = await response.json() as CopilotResponse;
    const output = Array.isArray(json.output) ? json.output : [];
    const results = extractCopilotCitations(output);

    return {
      query,
      results,
      textResponse: extractCopilotTextResponse(json),
      usage: {
        input_tokens: json.usage?.input_tokens ?? 0,
        output_tokens: json.usage?.output_tokens ?? 0,
        web_search_requests: resolveSearchRequestCount(json, output),
      },
    };
  } catch (error) {
    if ((error as Error)?.name === "AbortError") {
      throw new Error(`GitHub Copilot web search timed out after ${COPILOT_TIMEOUT_MS}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export const __testOnly = {
  COPILOT_RESPONSES_ENDPOINT,
  DEFAULT_COPILOT_SEARCH_MODEL,
  COPILOT_TIMEOUT_MS,
};
