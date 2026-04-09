import type { SearchAuth, SearchOptions, WebSearchResponse, WebSearchToolResult } from "./web-search";

const COPILOT_RESPONSES_ENDPOINT = "https://api.githubcopilot.com/responses";
/**
 * GitHub Copilot web search always uses this fixed known-good GPT-family model.
 * Magnitude intentionally ignores the selected Copilot slot model and options.model for web search.
 */
const FIXED_COPILOT_SEARCH_MODEL = "gpt-5.4";
const COPILOT_SEARCH_TOOL_ID = "copilot-search";
// Copilot Responses calls can legitimately run longer than 30s under load.
const COPILOT_REQUEST_TIMEOUT_MS = 90_000;

const COPILOT_HEADERS: Record<string, string> = {
  "Openai-Intent": "conversation-edits",
  "x-initiator": "user",
  "User-Agent": "GitHubCopilotChat/0.35.0",
  "Editor-Version": "vscode/1.107.0",
  "Editor-Plugin-Version": "copilot-chat/0.35.0",
  "Copilot-Integration-Id": "vscode-chat",
};

export function buildCopilotWebSearchRequest(
  query: string,
  options?: SearchOptions,
): Record<string, unknown> {
  return {
    model: FIXED_COPILOT_SEARCH_MODEL,
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
    stream: false,
  };
}

function normalizeSource(source: any): { title: string; url: string } | null {
  const url = typeof source === "string" ? source : source?.url;
  if (!url || typeof url !== "string") return null;
  const title =
    (typeof source === "object" && source && typeof source.title === "string" && source.title) || url;
  return { title, url };
}

export function extractCopilotCitations(output: any[]): WebSearchToolResult[] {
  const annotated: { title: string; url: string }[] = [];
  const seenAnnotated = new Set<string>();

  for (const item of output) {
    if (item?.type !== "message") continue;
    for (const content of item.content ?? []) {
      if (content?.type !== "output_text" || !Array.isArray(content.annotations)) continue;
      for (const annotation of content.annotations) {
        if (annotation?.type !== "url_citation" || typeof annotation.url !== "string") continue;
        if (seenAnnotated.has(annotation.url)) continue;
        seenAnnotated.add(annotation.url);
        annotated.push({ title: annotation.title ?? annotation.url, url: annotation.url });
      }
    }
  }

  if (annotated.length > 0) {
    return [{ tool_use_id: COPILOT_SEARCH_TOOL_ID, content: annotated }];
  }

  const fallback: { title: string; url: string }[] = [];
  const seenFallback = new Set<string>();
  for (const item of output) {
    if (item?.type !== "web_search_call" || !Array.isArray(item?.action?.sources)) continue;
    for (const rawSource of item.action.sources) {
      const source = normalizeSource(rawSource);
      if (!source || seenFallback.has(source.url)) continue;
      seenFallback.add(source.url);
      fallback.push(source);
    }
  }

  return fallback.length > 0
    ? [{ tool_use_id: COPILOT_SEARCH_TOOL_ID, content: fallback }]
    : [];
}

export function countCopilotSearchCalls(output: any[]): number {
  return output.filter((item: any) => item?.type === "web_search_call").length;
}

export function extractCopilotTextResponse(response: any): string {
  if (typeof response?.output_text === "string" && response.output_text.length > 0) {
    return response.output_text;
  }

  const chunks: string[] = [];
  for (const item of response?.output ?? []) {
    if (item?.type !== "message") continue;
    for (const content of item.content ?? []) {
      if (content?.type === "output_text" && typeof content.text === "string") {
        chunks.push(content.text);
      }
    }
  }
  return chunks.join("\n").trim();
}

function isCopilotTimeoutError(error: unknown): boolean {
  return error instanceof Error && /timed out after/i.test(error.message);
}

async function executeCopilotRequest(
  authToken: string,
  body: Record<string, unknown>,
): Promise<any> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), COPILOT_REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(COPILOT_RESPONSES_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${authToken}`,
        ...COPILOT_HEADERS,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      const error = new Error(`GitHub Copilot web search error ${response.status}: ${errorText}`) as Error & {
        status?: number;
        unauthorized?: boolean;
      };
      error.status = response.status;
      error.unauthorized =
        response.status === 401 &&
        /unauthorized|token expired|token_expired|expired/i.test(errorText);
      throw error;
    }

    return await response.json();
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`GitHub Copilot web search timed out after ${COPILOT_REQUEST_TIMEOUT_MS}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function copilotWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
  resolveRefreshedAuth?: () => Promise<SearchAuth | null>,
): Promise<WebSearchResponse> {
  if (auth.type !== "oauth-token" || !auth.value) {
    throw new Error("No GitHub Copilot OAuth session available for web search. Authenticate GitHub Copilot in the app.");
  }

  const body = buildCopilotWebSearchRequest(query, options);

  const executeWith401Retry = async (token: string): Promise<any> => {
    try {
      return await executeCopilotRequest(token, body);
    } catch (error) {
      const err = error as Error & { unauthorized?: boolean; status?: number };
      const shouldRetry = err.status === 401 || err.unauthorized === true;
      if (!shouldRetry || !resolveRefreshedAuth) throw error;

      const refreshedAuth = await resolveRefreshedAuth();
      if (!refreshedAuth || refreshedAuth.type !== "oauth-token" || !refreshedAuth.value) {
        throw new Error("GitHub Copilot web search unauthorized (401). Failed to refresh Copilot auth.");
      }

      try {
        return await executeCopilotRequest(refreshedAuth.value, body);
      } catch (retryError) {
        const retryErr = retryError as Error & { status?: number; unauthorized?: boolean };
        if (retryErr.status === 401 || retryErr.unauthorized === true) {
          throw new Error("GitHub Copilot web search unauthorized (401) after auth refresh retry.");
        }
        throw retryError;
      }
    }
  };

  let payload: any;
  try {
    payload = await executeWith401Retry(auth.value);
  } catch (error) {
    if (!isCopilotTimeoutError(error)) {
      throw error;
    }

    let retryToken = auth.value;
    if (resolveRefreshedAuth) {
      const refreshedForTimeoutRetry = await resolveRefreshedAuth().catch(() => null);
      if (
        refreshedForTimeoutRetry &&
        refreshedForTimeoutRetry.type === "oauth-token" &&
        refreshedForTimeoutRetry.value
      ) {
        retryToken = refreshedForTimeoutRetry.value;
      }
    }

    payload = await executeWith401Retry(retryToken);
  }

  const output = Array.isArray(payload?.output) ? payload.output : [];
  const textResponse = extractCopilotTextResponse(payload);
  const results = extractCopilotCitations(output);

  return {
    query,
    results,
    textResponse,
    usage: {
      input_tokens: payload?.usage?.input_tokens ?? 0,
      output_tokens: payload?.usage?.output_tokens ?? 0,
      web_search_requests: countCopilotSearchCalls(output),
    },
  };
}
