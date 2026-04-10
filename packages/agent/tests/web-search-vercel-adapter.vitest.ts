import { beforeEach, describe, expect, it, vi } from "vitest";

const generateTextMock = vi.fn();
const providerModelMock = vi.fn((modelId: string) => ({ id: modelId }));
const webSearchMock = vi.fn((_config?: unknown) => ({ type: "web_search_helper" }));
const webSearchPreviewMock = vi.fn((_config?: unknown) => ({ type: "web_search_preview_helper" }));
const createOpenAIMock = vi.fn((_config: { apiKey: string; baseURL?: string }) => {
  const provider = ((modelId: string) => providerModelMock(modelId)) as any;
  provider.responses = (modelId: string) => providerModelMock(modelId);
  provider.tools = {
    webSearch: webSearchMock,
  };
  return provider;
});

vi.mock("ai", () => ({
  generateText: generateTextMock,
}));

vi.mock("@ai-sdk/openai", () => ({
  createOpenAI: createOpenAIMock,
}));

describe("web-search vercel adapter", () => {
  beforeEach(() => {
    generateTextMock.mockReset();
    providerModelMock.mockReset();
    createOpenAIMock.mockClear();
    webSearchMock.mockReset();
    webSearchPreviewMock.mockReset();

    generateTextMock.mockResolvedValue({
      text: "answer",
      sources: [{ title: "Example", url: "https://example.com" }],
      usage: { inputTokens: 2, outputTokens: 4 },
      toolCalls: [{ toolName: "web_search" }],
    });
  });

  it("uses OpenAI webSearch helper (not provider-defined id) and normalizes response", async () => {
    const { vercelWebSearch } = await import("../src/tools/web-search-vercel");

    const result = await vercelWebSearch(
      "query",
      { type: "api-key", value: "vercel-key" },
      {
        system: "be concise",
        allowed_domains: ["example.com"],
        blocked_domains: ["reddit.com"],
        model: "openai/gpt-5.4",
      },
    );

    expect(createOpenAIMock).toHaveBeenCalledWith({
      apiKey: "vercel-key",
      baseURL: "https://ai-gateway.vercel.sh/v1",
    });
    expect(providerModelMock).toHaveBeenCalledWith("openai/gpt-5.4");
    expect(webSearchMock).toHaveBeenCalledWith({
      filters: { allowed_domains: ["example.com"] },
    });
    expect(webSearchPreviewMock).not.toHaveBeenCalled();
    expect(generateTextMock).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: "query",
        system: "be concise",
        tools: {
          web_search: { type: "web_search_helper" },
        },
      }),
    );
    expect(JSON.stringify(generateTextMock.mock.calls[0]?.[0] ?? {})).not.toContain("openai.web_search_preview");
    expect(result.results).toEqual([
      {
        tool_use_id: "vercel-search",
        content: [{ title: "Example", url: "https://example.com" }],
      },
    ]);
    expect(result.usage.web_search_requests).toBe(1);
  });

  it("throws explicit error when webSearch helper is unavailable", async () => {
    createOpenAIMock.mockImplementationOnce((_config: { apiKey: string; baseURL?: string }) => {
      const provider = ((modelId: string) => providerModelMock(modelId)) as any;
      provider.responses = (modelId: string) => providerModelMock(modelId);
      provider.tools = {
        webSearchPreview: webSearchPreviewMock,
      };
      return provider;
    });

    const { vercelWebSearch } = await import("../src/tools/web-search-vercel");

    await expect(
      vercelWebSearch("query", { type: "api-key", value: "vercel-key" }),
    ).rejects.toThrow("Vercel OpenAI webSearch helper not available in @ai-sdk/openai version");
  });

  it("throws explicit error when no OpenAI web search helper is available", async () => {
    createOpenAIMock.mockImplementationOnce((_config: { apiKey: string; baseURL?: string }) => {
      const provider = ((modelId: string) => providerModelMock(modelId)) as any;
      provider.responses = (modelId: string) => providerModelMock(modelId);
      provider.tools = {};
      return provider;
    });

    const { vercelWebSearch } = await import("../src/tools/web-search-vercel");

    await expect(
      vercelWebSearch("query", { type: "api-key", value: "vercel-key" }),
    ).rejects.toThrow("Vercel OpenAI webSearch helper not available in @ai-sdk/openai version");
  });
});
