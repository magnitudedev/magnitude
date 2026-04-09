import { describe, expect, mock, test } from "bun:test";
import {
  __testOnly,
  buildCopilotWebSearchRequest,
  copilotWebSearch,
  countCopilotSearchCalls,
  extractCopilotCitations,
  extractCopilotTextResponse,
} from "../web-search-copilot";

const originalFetch = globalThis.fetch;
const originalSetTimeout = globalThis.setTimeout;
const originalClearTimeout = globalThis.clearTimeout;

async function withFetchMock<T>(fetchImpl: typeof fetch, fn: () => Promise<T>): Promise<T> {
  (globalThis as any).fetch = fetchImpl;
  try {
    return await fn();
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

describe("web-search-copilot backend", () => {
  test("buildCopilotWebSearchRequest includes web_search tool and include sources", () => {
    expect(buildCopilotWebSearchRequest("latest AI news", {
      allowed_domains: ["github.com"],
      system: "be concise",
      model: "gpt-5.5-mini",
    })).toEqual({
      model: "gpt-5.5-mini",
      input: "latest AI news",
      stream: false,
      tools: [{ type: "web_search", filters: { allowed_domains: ["github.com"] } }],
      include: ["web_search_call.action.sources"],
      reasoning: { effort: "none" },
      instructions: "be concise",
    });
  });

  test("extracts annotation-only citations", () => {
    expect(extractCopilotCitations([
      {
        type: "message",
        content: [
          {
            type: "output_text",
            annotations: [
              { type: "url_citation", title: "A", url: "https://example.com/a" },
              { type: "url_citation", title: "B", url: "https://example.com/b" },
            ],
          },
        ],
      },
    ] as any)).toEqual([
      {
        tool_use_id: "copilot-search",
        content: [
          { title: "A", url: "https://example.com/a" },
          { title: "B", url: "https://example.com/b" },
        ],
      },
    ]);
  });

  test("falls back to web_search_call.action.sources when annotations are absent", () => {
    expect(extractCopilotCitations([
      {
        type: "web_search_call",
        action: { sources: ["https://example.com/a", { title: "B", url: "https://example.com/b" }] },
      },
    ] as any)).toEqual([
      {
        tool_use_id: "copilot-search",
        content: [
          { title: "https://example.com/a", url: "https://example.com/a" },
          { title: "B", url: "https://example.com/b" },
        ],
      },
    ]);
  });

  test("merges and dedupes annotation + source citations with annotation title precedence", () => {
    expect(extractCopilotCitations([
      {
        type: "message",
        content: [
          {
            type: "output_text",
            annotations: [
              { type: "url_citation", title: "Annotated A", url: "https://example.com/a" },
            ],
          },
        ],
      },
      {
        type: "web_search_call",
        action: {
          sources: [
            { title: "Source A", url: "https://example.com/a" },
            { title: "Source B", url: "https://example.com/b" },
          ],
        },
      },
    ] as any)).toEqual([
      {
        tool_use_id: "copilot-search",
        content: [
          { title: "Annotated A", url: "https://example.com/a" },
          { title: "Source B", url: "https://example.com/b" },
        ],
      },
    ]);
  });

  test("extractCopilotTextResponse falls back from output_text to message output_text content", () => {
    expect(extractCopilotTextResponse({
      output: [
        { type: "message", content: [{ type: "output_text", text: "first" }, { type: "output_text", text: "second" }] },
      ],
    } as any)).toBe("first\nsecond");
  });

  test("countCopilotSearchCalls counts web_search_call items", () => {
    expect(countCopilotSearchCalls([
      { type: "message" },
      { type: "web_search_call" },
      { type: "web_search_call" },
    ] as any)).toBe(2);
  });

  test("sends direct POST /responses with Copilot auth + headers and normalizes usage/results", async () => {
    let capturedUrl = "";
    let capturedInit: RequestInit | undefined;

    await withFetchMock(
      (async (url: string | URL | Request, init?: RequestInit) => {
        capturedUrl = String(url);
        capturedInit = init;
        return new Response(JSON.stringify({
          output_text: "copilot answer",
          output: [
            {
              type: "message",
              content: [
                {
                  type: "output_text",
                  text: "copilot answer",
                  annotations: [{ type: "url_citation", title: "Docs", url: "https://docs.example.com" }],
                },
              ],
            },
            {
              type: "web_search_call",
              action: { sources: [{ url: "https://docs.example.com" }, { url: "https://extra.example.com" }] },
            },
          ],
          usage: { input_tokens: 10, output_tokens: 20 },
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }) as any,
      async () => {
        const result = await copilotWebSearch("query", { type: "oauth-token", value: "copilot-token" }, {
          system: "be concise",
          allowed_domains: ["github.com"],
        });

        expect(result).toEqual({
          query: "query",
          results: [
            {
              tool_use_id: "copilot-search",
              content: [
                { title: "Docs", url: "https://docs.example.com" },
                { title: "https://extra.example.com", url: "https://extra.example.com" },
              ],
            },
          ],
          textResponse: "copilot answer",
          usage: {
            input_tokens: 10,
            output_tokens: 20,
            web_search_requests: 1,
          },
        });
      },
    );

    expect(capturedUrl).toBe(__testOnly.COPILOT_RESPONSES_ENDPOINT);
    expect(capturedInit?.method).toBe("POST");
    expect(capturedInit?.headers).toMatchObject({
      Authorization: "Bearer copilot-token",
      "Content-Type": "application/json",
      "Openai-Intent": "conversation-edits",
      "x-initiator": "user",
      "User-Agent": "GitHubCopilotChat/0.35.0",
      "Editor-Version": "vscode/1.107.0",
      "Editor-Plugin-Version": "copilot-chat/0.35.0",
      "Copilot-Integration-Id": "vscode-chat",
    });

    expect(JSON.parse(String(capturedInit?.body))).toEqual({
      model: "gpt-5.4",
      input: "query",
      stream: false,
      tools: [{ type: "web_search", filters: { allowed_domains: ["github.com"] } }],
      include: ["web_search_call.action.sources"],
      reasoning: { effort: "none" },
      instructions: "be concise",
    });
  });

  test("falls back usage.web_search_requests when output has no web_search_call items", async () => {
    await withFetchMock(
      (async () => new Response(JSON.stringify({
        output_text: "copilot answer",
        output: [],
        usage: { input_tokens: 1, output_tokens: 2, web_search_requests: 3 },
      }), { status: 200 })) as any,
      async () => {
        const result = await copilotWebSearch("query", { type: "oauth-token", value: "copilot-token" });
        expect(result.usage).toEqual({
          input_tokens: 1,
          output_tokens: 2,
          web_search_requests: 3,
        });
      },
    );
  });

  test("surfaces non-2xx failures with status and body", async () => {
    await withFetchMock(
      (async () => new Response("bad gateway", { status: 502 })) as any,
      async () => {
        await expect(
          copilotWebSearch("query", { type: "oauth-token", value: "copilot-token" }),
        ).rejects.toThrow("GitHub Copilot web search error 502: bad gateway");
      },
    );
  });

  test("surfaces timeout errors clearly", async () => {
    const clearTimeoutMock = mock((_id: any) => undefined);
    (globalThis as any).clearTimeout = clearTimeoutMock;
    (globalThis as any).setTimeout = ((_fn: () => void) => {
      _fn();
      return 1 as any;
    }) as typeof setTimeout;

    try {
      await withFetchMock(
        (async (_url: string | URL | Request, init?: RequestInit) => {
          const abortError = new Error("aborted");
          (abortError as any).name = "AbortError";
          if ((init?.signal as AbortSignal | undefined)?.aborted) throw abortError;
          throw abortError;
        }) as any,
        async () => {
          await expect(
            copilotWebSearch("query", { type: "oauth-token", value: "copilot-token" }),
          ).rejects.toThrow(`GitHub Copilot web search timed out after ${__testOnly.COPILOT_TIMEOUT_MS}ms`);
        },
      );
    } finally {
      (globalThis as any).setTimeout = originalSetTimeout;
      (globalThis as any).clearTimeout = originalClearTimeout;
    }
  });

});
