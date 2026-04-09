import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildCopilotWebSearchRequest,
  copilotWebSearch,
  countCopilotSearchCalls,
  extractCopilotCitations,
  extractCopilotTextResponse,
  isCopilotResponsesSearchCapable,
  selectCopilotSearchModel,
} from "./web-search-copilot";

describe("web-search-copilot", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("builds a Copilot responses request with headers and web_search tool", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        output_text: "answer",
        output: [],
        usage: { input_tokens: 1, output_tokens: 2 },
      }),
    });

    vi.stubGlobal("fetch", fetchMock);

    await copilotWebSearch(
      "latest AI news",
      { type: "oauth-token", value: "copilot-token" },
      { system: "be concise", allowed_domains: ["github.com"], model: "gpt-5" },
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://api.githubcopilot.com/responses");
    expect(init.method).toBe("POST");
    expect(init.headers).toMatchObject({
      "Content-Type": "application/json",
      "Authorization": "Bearer copilot-token",
      "Openai-Intent": "conversation-edits",
      "x-initiator": "user",
      "User-Agent": "GitHubCopilotChat/0.35.0",
      "Editor-Version": "vscode/1.107.0",
      "Editor-Plugin-Version": "copilot-chat/0.35.0",
      "Copilot-Integration-Id": "vscode-chat",
    });

    const body = JSON.parse(init.body);
    expect(body).toEqual({
      model: "gpt-5",
      input: "latest AI news",
      tools: [{ type: "web_search", filters: { allowed_domains: ["github.com"] } }],
      include: ["web_search_call.action.sources"],
      instructions: "be concise",
    });
  });

  it("prefers url_citation annotations and deduplicates urls", () => {
    const results = extractCopilotCitations([
      {
        type: "message",
        content: [
          {
            type: "output_text",
            text: "hello",
            annotations: [
              { type: "url_citation", title: "One", url: "https://example.com/1" },
              { type: "url_citation", title: "One duplicate", url: "https://example.com/1" },
              { type: "url_citation", title: "Two", url: "https://example.com/2" },
            ],
          },
        ],
      },
      {
        type: "web_search_call",
        action: {
          sources: [{ title: "Ignored fallback", url: "https://fallback.example.com" }],
        },
      },
    ]);

    expect(results).toEqual([
      {
        tool_use_id: "copilot-search",
        content: [
          { title: "One", url: "https://example.com/1" },
          { title: "Two", url: "https://example.com/2" },
        ],
      },
    ]);
  });

  it("falls back to web_search_call sources when annotations are absent", () => {
    const results = extractCopilotCitations([
      {
        type: "web_search_call",
        action: {
          sources: [
            { title: "Source One", url: "https://example.com/1" },
            { title: "Source One Duplicate", url: "https://example.com/1" },
            "https://example.com/2",
          ],
        },
      },
    ]);

    expect(results).toEqual([
      {
        tool_use_id: "copilot-search",
        content: [
          { title: "Source One", url: "https://example.com/1" },
          { title: "https://example.com/2", url: "https://example.com/2" },
        ],
      },
    ]);
  });

  it("extracts text and usage from a Copilot response", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        output: [
          {
            type: "message",
            content: [
              {
                type: "output_text",
                text: "First paragraph.",
                annotations: [{ type: "url_citation", title: "Docs", url: "https://docs.example.com" }],
              },
              {
                type: "output_text",
                text: "Second paragraph.",
              },
            ],
          },
          { type: "web_search_call", action: { sources: [{ url: "https://ignored.example.com" }] } },
        ],
        usage: { input_tokens: 11, output_tokens: 29 },
      }),
    });

    vi.stubGlobal("fetch", fetchMock);

    const result = await copilotWebSearch(
      "query",
      { type: "oauth-token", value: "copilot-token" },
      { model: "gpt-5" },
    );

    expect(result).toEqual({
      query: "query",
      results: [
        {
          tool_use_id: "copilot-search",
          content: [{ title: "Docs", url: "https://docs.example.com" }],
        },
      ],
      textResponse: "First paragraph.\nSecond paragraph.",
      usage: {
        input_tokens: 11,
        output_tokens: 29,
        web_search_requests: 1,
      },
    });
  });

  it("falls back to a known-safe model for unsupported active Copilot models", () => {
    expect(isCopilotResponsesSearchCapable("gpt-5")).toBe(true);
    expect(isCopilotResponsesSearchCapable("gpt-6-codex")).toBe(true);
    expect(isCopilotResponsesSearchCapable("gpt-5-mini")).toBe(false);
    expect(isCopilotResponsesSearchCapable("claude-sonnet-4.6")).toBe(false);

    expect(selectCopilotSearchModel("gpt-5")).toBe("gpt-5");
    expect(selectCopilotSearchModel("claude-sonnet-4.6")).toBe("gpt-5");
    expect(buildCopilotWebSearchRequest("q", { model: "claude-sonnet-4.6" })).toMatchObject({
      model: "gpt-5",
    });
  });

  it("throws clearly when no Copilot OAuth token is available", async () => {
    await expect(
      copilotWebSearch("query", { type: "api-key", value: "not-allowed" }),
    ).rejects.toThrow("No GitHub Copilot OAuth session available for web search");
  });

  it("exposes small parsing helpers for fallback behavior", () => {
    expect(countCopilotSearchCalls([{ type: "web_search_call" }, { type: "message" }])).toBe(1);
    expect(
      extractCopilotTextResponse({
        output: [
          {
            type: "message",
            content: [
              { type: "output_text", text: "A" },
              { type: "output_text", text: "B" },
            ],
          },
        ],
      }),
    ).toBe("A\nB");
  });
});
