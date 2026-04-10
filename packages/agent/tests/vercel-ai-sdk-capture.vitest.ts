import { describe, expect, it } from "vitest";
import { __testOnly } from "../scripts/web-search-capture/vercel-ai-sdk-capture";

describe("vercel ai sdk capture helper", () => {
  it("normalizes url-based sources and deduplicates by URL", () => {
    const normalized = __testOnly.normalizeSources([
      { title: "A", url: "https://a.test" },
      { title: "B", href: "https://b.test" },
      { title: "A duplicate", url: "https://a.test" },
      { title: "No URL" },
    ]);

    expect(normalized).toEqual([
      { title: "A", url: "https://a.test" },
      { title: "B", url: "https://b.test" },
    ]);
  });

  it("extracts warning types and detects unsupported-tool", () => {
    const warnings = [
      { type: "unsupported-setting", setting: "temperature" },
      { type: "unsupported-tool", tool: { id: "web_search" } },
    ];
    expect(__testOnly.parseWarningTypes(warnings)).toEqual([
      "unsupported-setting",
      "unsupported-tool",
    ]);
    expect(__testOnly.hasUnsupportedToolWarning(warnings)).toBe(true);
  });

  it("detects when downstream request body dropped tools", () => {
    const droppedBody = JSON.stringify({ model: "openai/gpt-5.4", tools: [] });
    const preservedBody = JSON.stringify({ model: "openai/gpt-5.4", tools: [{ type: "web_search" }] });

    expect(__testOnly.requestBodyHasEmptyTools(droppedBody)).toBe(true);
    expect(__testOnly.requestBodyHasEmptyTools(preservedBody)).toBe(false);
    expect(__testOnly.requestBodyHasEmptyTools("not-json")).toBe(false);
  });

  it("requires webSearch helper", () => {
    const providerWithWebSearch = {
      tools: {
        webSearch: (config?: unknown) => ({ kind: "webSearch", config }),
      },
    } as any;

    const resolved = __testOnly.resolveOpenAIWebSearchTool(providerWithWebSearch, {
      allowed_domains: ["example.com"],
    });
    expect(resolved.helper).toBe("webSearch");
    expect(resolved.tool).toMatchObject({
      kind: "webSearch",
      config: { filters: { allowed_domains: ["example.com"] } },
    });
  });

  it("throws explicit error when webSearch helper is unavailable", () => {
    expect(() =>
      __testOnly.resolveOpenAIWebSearchTool({
        tools: {
          webSearchPreview: (config?: unknown) => ({ kind: "webSearchPreview", config }),
        },
      } as any)
    ).toThrow("Vercel OpenAI webSearch helper not available in @ai-sdk/openai version");
  });
});
