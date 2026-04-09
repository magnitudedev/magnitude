import { describe, expect, it } from "vitest";
import { __testOnly } from "../scripts/web-search-capture";

describe("web-search-capture vercel special-case status", () => {
  it("marks capture-error when tool is unsupported and dropped", () => {
    const assessed = __testOnly.getVercelSpecialCaseStatusAndNotes({
      diagnostics: {
        unsupportedToolWarning: true,
        requestToolsDropped: true,
        requestedWebSearchTool: true,
        hasCitations: false,
        warningTypes: ["unsupported-setting", "unsupported-tool"],
      },
    });

    expect(assessed.status).toBe("capture-error");
    expect(assessed.notes).toContain("vercel:unsupported-tool-warning");
    expect(assessed.notes).toContain("vercel:request-tools-dropped");
    expect(assessed.notes).toContain("vercel:missing-structured-sources");
  });

  it("keeps success only when request/tools/sources/usage evidence is present", () => {
    const assessed = __testOnly.getVercelSpecialCaseStatusAndNotes({
      diagnostics: {
        unsupportedToolWarning: false,
        requestToolsDropped: false,
        requestedWebSearchTool: true,
        hasCitations: true,
        warningTypes: [],
      },
      request: {
        args: {
          tools: {
            web_search: { type: "web_search" },
          },
        },
      },
      normalizedResult: {
        results: [
          {
            content: [{ title: "Example", url: "https://example.com" }],
          },
        ],
        usage: { web_search_requests: 1 },
      },
    });

    expect(assessed.status).toBe("success");
    expect(assessed.notes).toEqual([]);
  });

  it("marks capture-error when vercel evidence is incomplete", () => {
    const assessed = __testOnly.getVercelSpecialCaseStatusAndNotes({
      diagnostics: {
        unsupportedToolWarning: false,
        requestToolsDropped: false,
        requestedWebSearchTool: true,
        hasCitations: true,
        warningTypes: [],
      },
      request: {
        args: {
          tools: {},
        },
      },
      normalizedResult: {
        results: [],
        usage: { web_search_requests: 0 },
      },
    });

    expect(assessed.status).toBe("capture-error");
    expect(assessed.notes).toContain("vercel:missing-tool-payload");
    expect(assessed.notes).toContain("vercel:missing-structured-sources");
    expect(assessed.notes).toContain("vercel:web-search-requests-lt-1");
  });
});
