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
});
