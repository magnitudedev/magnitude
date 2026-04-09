import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

const originalEnv = {
  MAGNITUDE_SEARCH_PROVIDER: process.env.MAGNITUDE_SEARCH_PROVIDER,
  AI_GATEWAY_API_KEY: process.env.AI_GATEWAY_API_KEY,
  VERCEL_API_KEY: process.env.VERCEL_API_KEY,
};

const createResponseMock = mock(async (_request: unknown) => ({
  output: [],
  output_text: "",
  usage: { input_tokens: 0, output_tokens: 0 },
}));

mock.module("openai", () => {
  class MockOpenAI {
    responses: { create: typeof createResponseMock };

    constructor(_config: { apiKey: string; baseURL?: string }) {
      this.responses = { create: createResponseMock };
    }
  }

  return { default: MockOpenAI };
});

const { detectSearchProvider, resolveVercelAuth } = await import("./web-search");
const {
  countVercelSearchCalls,
  extractVercelCitations,
  vercelWebSearch,
} = await import("./web-search-vercel");

beforeEach(() => {
  process.env.MAGNITUDE_SEARCH_PROVIDER = originalEnv.MAGNITUDE_SEARCH_PROVIDER;
  process.env.AI_GATEWAY_API_KEY = originalEnv.AI_GATEWAY_API_KEY;
  process.env.VERCEL_API_KEY = originalEnv.VERCEL_API_KEY;
  createResponseMock.mockReset();
  createResponseMock.mockImplementation(async (_request: unknown) => ({
    output: [],
    output_text: "",
    usage: { input_tokens: 0, output_tokens: 0 },
  }));
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
  test("extracts url_citation annotations with deduplication", () => {
    const result = extractVercelCitations([
      {
        type: "message",
        content: [
          {
            type: "output_text",
            annotations: [
              { type: "url_citation", title: "Example", url: "https://example.com" },
              { type: "url_citation", title: "Example duplicate", url: "https://example.com" },
              { type: "other", title: "Ignore", url: "https://ignored.com" },
            ],
          },
        ],
      },
    ]);

    expect(result).toEqual([
      {
        tool_use_id: "vercel-search",
        content: [{ title: "Example", url: "https://example.com" }],
      },
    ]);
  });

  test("falls back to web_search_call sources and preserves titles when present", () => {
    const result = extractVercelCitations([
      {
        type: "web_search_call",
        action: {
          sources: [
            { title: "Alpha", url: "https://alpha.test" },
            "https://beta.test",
            { title: "Alpha duplicate", url: "https://alpha.test" },
          ],
        },
      },
    ]);

    expect(result).toEqual([
      {
        tool_use_id: "vercel-search",
        content: [
          { title: "Alpha", url: "https://alpha.test" },
          { title: "https://beta.test", url: "https://beta.test" },
        ],
      },
    ]);
  });

  test("prefers url_citation annotations over fallback action sources", () => {
    const result = extractVercelCitations([
      {
        type: "message",
        content: [
          {
            type: "output_text",
            annotations: [
              { type: "url_citation", title: "Canonical", url: "https://example.com" },
            ],
          },
        ],
      },
      {
        type: "web_search_call",
        action: {
          sources: [{ title: "Ignored fallback", url: "https://fallback.test" }],
        },
      },
    ]);

    expect(result).toEqual([
      {
        tool_use_id: "vercel-search",
        content: [{ title: "Canonical", url: "https://example.com" }],
      },
    ]);
  });

  test("counts web search calls", () => {
    expect(
      countVercelSearchCalls([
        { type: "message" },
        { type: "web_search_call" },
        { type: "web_search_call" },
      ]),
    ).toBe(2);
  });

  test("normalizes a successful Vercel response and always uses fixed gpt-5.4", async () => {
    createResponseMock.mockImplementation(async () => ({
      output: [
        {
          type: "message",
          content: [
            {
              type: "output_text",
              annotations: [
                { type: "url_citation", title: "Example", url: "https://example.com" },
              ],
            },
          ],
        },
        {
          type: "web_search_call",
          action: {
            sources: [{ title: "Ignored by annotation precedence", url: "https://example.com" }],
          },
        },
      ],
      output_text: "Answer text",
      usage: { input_tokens: 21, output_tokens: 9 },
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

    expect(createResponseMock.mock.calls[0]?.[0]).toMatchObject({
      model: "openai/gpt-5.4",
      input: "query",
      instructions: "Be concise",
      include: ["web_search_call.action.sources"],
      tools: [
        {
          type: "web_search",
          filters: { allowed_domains: ["example.com"] },
        },
      ],
    });
  });

  test("ignores options.model for Vercel search requests", async () => {
    await vercelWebSearch("query", { type: "api-key", value: "vercel-key" }, {
      model: "anthropic/claude-haiku-4.5",
    });

    expect(createResponseMock.mock.calls[0]?.[0]).toMatchObject({
      model: "openai/gpt-5.4",
    });
  });

  test("handles responses with no citations gracefully", async () => {
    createResponseMock.mockImplementation(async () => ({
      output: [{ type: "message", content: [{ type: "output_text", annotations: [] }] }],
      output_text: "No sources found",
      usage: { input_tokens: 5, output_tokens: 3 },
    }));

    const result = await vercelWebSearch("query", { type: "api-key", value: "vercel-key" });

    expect(result.results).toEqual([]);
    expect(result.textResponse).toBe("No sources found");
    expect(result.usage.web_search_requests).toBe(0);
  });
});
