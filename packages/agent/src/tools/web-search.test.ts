import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

const originalEnv = {
  MAGNITUDE_SEARCH_PROVIDER: process.env.MAGNITUDE_SEARCH_PROVIDER,
  AI_GATEWAY_API_KEY: process.env.AI_GATEWAY_API_KEY,
  VERCEL_API_KEY: process.env.VERCEL_API_KEY,
};

const generateTextMock = mock(async (_request: unknown) => ({
  sources: [],
  text: "",
  usage: { inputTokens: 0, outputTokens: 0 },
}));

const perplexitySearchMock = mock((_options?: unknown) => ({ type: "perplexity-search-tool", _options }));
const gatewayTools = { perplexitySearch: perplexitySearchMock };
const providerModelMock = mock((_modelId: string) => ({ id: _modelId }));
const createOpenAIMock = mock((_config: { apiKey: string; baseURL?: string }) => {
  const provider = ((modelId: string) => providerModelMock(modelId)) as any;
  return provider;
});

mock.module("ai", () => ({ generateText: generateTextMock }));
mock.module("@ai-sdk/gateway", () => ({ gateway: { tools: gatewayTools } }));
mock.module("@ai-sdk/openai", () => ({ createOpenAI: createOpenAIMock }));

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
  generateTextMock.mockImplementation(async (_request: unknown) => ({
    sources: [],
    text: "",
    usage: { inputTokens: 0, outputTokens: 0 },
  }));

  perplexitySearchMock.mockReset();
  perplexitySearchMock.mockImplementation((_options?: unknown) => ({ type: "perplexity-search-tool", _options }));

  providerModelMock.mockReset();
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

  test("normalizes a successful Vercel AI SDK response and uses gateway perplexity search tool", async () => {
    generateTextMock.mockImplementation(async () => ({
      sources: [
        { title: "Example", url: "https://example.com" },
        { title: "Duplicate", url: "https://example.com" },
      ],
      text: "Answer text",
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

    expect(providerModelMock.mock.calls[0]?.[0]).toBe("anthropic/claude-sonnet-4.6");
    expect(generateTextMock.mock.calls[0]?.[0]).toMatchObject({
      prompt: "query",
      system: "Be concise",
      tools: {
        perplexity_search: { type: "perplexity-search-tool", _options: { searchDomainFilter: ["example.com"] } },
      },
    });

    expect(perplexitySearchMock.mock.calls[0]?.[0]).toEqual({ searchDomainFilter: ["example.com"] });
  });

  test("defaults model when options.model is not provided", async () => {
    await vercelWebSearch("query", { type: "api-key", value: "vercel-key" });

    expect(providerModelMock.mock.calls[0]?.[0]).toBe("openai/gpt-5.4");
  });

  test("maps blocked domains to deny-list searchDomainFilter", async () => {
    await vercelWebSearch("query", { type: "api-key", value: "vercel-key" }, {
      blocked_domains: ["reddit.com", "x.com"],
    });

    expect(perplexitySearchMock.mock.calls[0]?.[0]).toEqual({
      searchDomainFilter: ["-reddit.com", "-x.com"],
    });
  });

  test("prefers allowed domains when both allowed and blocked are provided", async () => {
    await vercelWebSearch("query", { type: "api-key", value: "vercel-key" }, {
      allowed_domains: ["example.com"],
      blocked_domains: ["reddit.com"],
    });

    expect(perplexitySearchMock.mock.calls[0]?.[0]).toEqual({
      searchDomainFilter: ["example.com"],
    });
  });

  test("handles responses with no citations gracefully", async () => {
    generateTextMock.mockImplementation(async () => ({
      sources: [],
      text: "No sources found",
      usage: { inputTokens: 5, outputTokens: 3 },
    }));

    const result = await vercelWebSearch("query", { type: "api-key", value: "vercel-key" });

    expect(result.results).toEqual([]);
    expect(result.textResponse).toBe("No sources found");
    expect(result.usage.web_search_requests).toBe(0);
  });

  test("falls back to step and tool-result sources when top-level sources are empty", async () => {
    generateTextMock.mockImplementation(async () => ({
      sources: [],
      steps: [
        {
          sources: [{ title: "From step", url: "https://step.example" }],
          toolResults: [{ sources: [{ title: "From tool", url: "https://tool.example" }] }],
        },
      ],
      text: "Answer with sources",
      usage: { inputTokens: 8, outputTokens: 4 },
    }));

    const result = await vercelWebSearch("query", { type: "api-key", value: "vercel-key" });

    expect(result.results).toEqual([
      {
        tool_use_id: "vercel-search",
        content: [
          { title: "From step", url: "https://step.example" },
          { title: "From tool", url: "https://tool.example" },
        ],
      },
    ]);
    expect(result.usage.web_search_requests).toBe(1);
  });
});
