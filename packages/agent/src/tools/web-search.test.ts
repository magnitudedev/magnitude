import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

const originalEnv = {
  MAGNITUDE_SEARCH_PROVIDER: process.env.MAGNITUDE_SEARCH_PROVIDER,
  AI_GATEWAY_API_KEY: process.env.AI_GATEWAY_API_KEY,
  VERCEL_API_KEY: process.env.VERCEL_API_KEY,
};

const generateTextMock = mock(async (_args: unknown) => ({
  text: "",
  usage: { inputTokens: 0, outputTokens: 0 },
}));

const webSearchMock = mock((config?: unknown) => ({ type: "web_search", ...(config as object) }));

const responsesMock = mock((model: string) => ({ providerModel: model }));

const createOpenAIMock = mock((config: { apiKey: string; baseURL?: string }) => ({
  config,
  responses: responsesMock,
  tools: {
    webSearch: webSearchMock,
  },
}));

mock.module("ai", () => ({
  __esModule: true,
  generateText: generateTextMock,
}));

mock.module("@ai-sdk/openai", () => ({
  __esModule: true,
  createOpenAI: createOpenAIMock,
}));

const { detectSearchProvider, resolveVercelAuth } = await import("./web-search");
const {
  extractVercelSources,
  vercelWebSearch,
} = await import("./web-search-vercel");

beforeEach(() => {
  process.env.MAGNITUDE_SEARCH_PROVIDER = originalEnv.MAGNITUDE_SEARCH_PROVIDER;
  process.env.AI_GATEWAY_API_KEY = originalEnv.AI_GATEWAY_API_KEY;
  process.env.VERCEL_API_KEY = originalEnv.VERCEL_API_KEY;
  generateTextMock.mockReset();
  generateTextMock.mockImplementation(async (_args: unknown) => ({
    text: "",
    usage: { inputTokens: 0, outputTokens: 0 },
  }));

  webSearchMock.mockReset();
  webSearchMock.mockImplementation((config?: unknown) => ({ type: "web_search", ...(config as object) }));

  responsesMock.mockReset();
  responsesMock.mockImplementation((model: string) => ({ providerModel: model }));

  createOpenAIMock.mockClear();
});

afterEach(() => {
  process.env.MAGNITUDE_SEARCH_PROVIDER = originalEnv.MAGNITUDE_SEARCH_PROVIDER;
  process.env.AI_GATEWAY_API_KEY = originalEnv.AI_GATEWAY_API_KEY;
  process.env.VERCEL_API_KEY = originalEnv.VERCEL_API_KEY;
});

describe("web-search vercel router reconciliation", () => {
  test("accepts MAGNITUDE_SEARCH_PROVIDER=vercel", () => {
    process.env.MAGNITUDE_SEARCH_PROVIDER = "vercel";
    expect(detectSearchProvider("openai")).toBe("vercel");
  });

  test("detects vercel as its own explicit backend identity", () => {
    delete process.env.MAGNITUDE_SEARCH_PROVIDER;
    expect(detectSearchProvider("vercel")).toBe("vercel");
  });

  test("invalid override list includes vercel", () => {
    process.env.MAGNITUDE_SEARCH_PROVIDER = "bogus";
    expect(() => detectSearchProvider("openai")).toThrow(/vercel/);
  });

  test("vercel override wins over slot provider identity", () => {
    process.env.MAGNITUDE_SEARCH_PROVIDER = "vercel";
    expect(detectSearchProvider("github-copilot")).toBe("vercel");
  });
});

describe("web-search vercel auth resolution", () => {
  test("prefers stored vercel auth over env", () => {
    process.env.AI_GATEWAY_API_KEY = "env-key";
    expect(resolveVercelAuth({ type: "api", key: "stored-key" } as any)).toEqual({
      type: "api-key",
      value: "stored-key",
    });
  });

  test("falls back to AI_GATEWAY_API_KEY", () => {
    process.env.AI_GATEWAY_API_KEY = "env-key";
    expect(resolveVercelAuth()).toEqual({ type: "api-key", value: "env-key" });
  });

  test("falls back to VERCEL_API_KEY alias when standard env var is absent", () => {
    delete process.env.AI_GATEWAY_API_KEY;
    process.env.VERCEL_API_KEY = "alias-key";
    expect(resolveVercelAuth()).toEqual({ type: "api-key", value: "alias-key" });
  });

  test("throws a clear Vercel-specific error when auth is missing", () => {
    delete process.env.AI_GATEWAY_API_KEY;
    delete process.env.VERCEL_API_KEY;
    expect(() => resolveVercelAuth()).toThrow(
      "No Vercel AI Gateway API key available for web search. Set AI_GATEWAY_API_KEY or authenticate Vercel in the app.",
    );
  });
});

describe("web-search-vercel normalization", () => {
  test("extracts normalized AI SDK sources with deduplication", () => {
    const result = extractVercelSources([
      { title: "Example", url: "https://example.com" },
      { title: "Example duplicate", url: "https://example.com" },
      { title: "Href fallback", href: "https://href.test" },
      { title: "No URL" },
    ]);

    expect(result).toEqual([
      {
        tool_use_id: "vercel-search",
        content: [
          { title: "Example", url: "https://example.com" },
          { title: "Href fallback", url: "https://href.test" },
        ],
      },
    ]);
  });

  test("normalizes a successful Vercel AI SDK response and uses OpenAI web_search tool", async () => {
    generateTextMock.mockImplementation(async () => ({
      text: "Answer text",
      sources: [
        { title: "Example", url: "https://example.com" },
        { title: "Duplicate", url: "https://example.com" },
      ],
      toolCalls: [{ type: "tool-call", toolName: "web_search" }],
      usage: { inputTokens: 21, outputTokens: 9 },
    }));

    const result = await vercelWebSearch("query", { type: "api-key", value: "vercel-key" }, {
      system: "Be concise",
      allowed_domains: ["example.com"],
      model: "anthropic/claude-sonnet-4.6",
    });

    expect(result).toEqual({
      query: "query",
      results: [
        {
          tool_use_id: "vercel-search",
          content: [{ title: "Example", url: "https://example.com" }],
        },
      ],
      textResponse: "Answer text",
      usage: {
        input_tokens: 21,
        output_tokens: 9,
        web_search_requests: 1,
      },
    });

    expect(createOpenAIMock.mock.calls[0]?.[0]).toEqual({
      apiKey: "vercel-key",
      baseURL: "https://ai-gateway.vercel.sh/v1",
    });

    expect(responsesMock.mock.calls[0]?.[0]).toBe("anthropic/claude-sonnet-4.6");
    expect(webSearchMock.mock.calls[0]?.[0]).toEqual({
      filters: { allowed_domains: ["example.com"] },
    });
    expect(generateTextMock.mock.calls[0]?.[0]).toMatchObject({
      model: { providerModel: "anthropic/claude-sonnet-4.6" },
      prompt: "query",
      system: "Be concise",
      tools: {
        web_search: {
          type: "web_search",
          filters: { allowed_domains: ["example.com"] },
        },
      },
      providerOptions: {
        openai: {
          forceReasoning: true,
          reasoningEffort: "none",
        },
      },
    });
  });

  test("defaults model when options.model is not provided", async () => {
    await vercelWebSearch("query", { type: "api-key", value: "vercel-key" });

    expect(responsesMock.mock.calls[0]?.[0]).toBe("openai/gpt-5.4");
  });

  test("passes allowed domains through OpenAI web_search filters", async () => {
    await vercelWebSearch("query", { type: "api-key", value: "vercel-key" }, {
      allowed_domains: ["example.com"],
      blocked_domains: ["reddit.com"],
    });

    expect(webSearchMock.mock.calls[0]?.[0]).toEqual({
      filters: { allowed_domains: ["example.com"] },
    });
  });

  test("does not add domain filters when only blocked domains are provided", async () => {
    await vercelWebSearch("query", { type: "api-key", value: "vercel-key" }, {
      blocked_domains: ["reddit.com", "x.com"],
    });

    expect(webSearchMock.mock.calls[0]?.[0]).toEqual({});
  });

  test("handles responses with no citations gracefully", async () => {
    generateTextMock.mockImplementation(async () => ({
      text: "No sources found",
      usage: { inputTokens: 5, outputTokens: 3 },
    }));

    const result = await vercelWebSearch("query", { type: "api-key", value: "vercel-key" });

    expect(result.results).toEqual([]);
    expect(result.textResponse).toBe("No sources found");
    expect(result.usage.web_search_requests).toBe(0);
  });

  test("merges top-level, step/tool-result, and markdown-link sources with stable URL dedupe", async () => {
    generateTextMock.mockImplementation(async () => ({
      text: "Answer with [Top Example](https://top.example) and [Extra Link](https://text.example).",
      sources: [{ title: "Top Citation", url: "https://top.example" }],
      steps: [
        {
          sources: [{ title: "Step Direct Source", url: "https://step-direct.example" }],
          toolResults: [
            {
              sources: [
                { title: "Step Duplicate Title", url: "https://top.example" },
                { title: "Step Source", url: "https://step.example" },
              ],
            },
          ],
        },
      ],
      toolCalls: [{ type: "tool-call", toolName: "web_search" }],
      usage: { inputTokens: 8, outputTokens: 4 },
    }));

    const result = await vercelWebSearch("query", { type: "api-key", value: "vercel-key" });

    expect(result.results).toEqual([
      {
        tool_use_id: "vercel-search",
        content: [
          { title: "Top Citation", url: "https://top.example" },
          { title: "Step Direct Source", url: "https://step-direct.example" },
          { title: "Step Source", url: "https://step.example" },
          { title: "Extra Link", url: "https://text.example" },
        ],
      },
    ]);
    expect(result.usage.web_search_requests).toBe(1);
  });
});
