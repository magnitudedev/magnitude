import { describe, expect, it } from "vitest";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import OpenAI from "openai";
import {
  sentinel,
  writeIndex,
  writeRunArtifacts,
} from "../scripts/web-search-capture/capture-harness";
import { withFetchInterceptor } from "../scripts/web-search-capture/interceptors/fetch-interceptor";
import { withOpenAISdkInterceptor } from "../scripts/web-search-capture/interceptors/openai-sdk-interceptor";

describe("web-search capture harness", () => {
  it("writes complete artifact file set and index", async () => {
    const root = await mkdtemp(join(tmpdir(), "web-search-capture-"));
    const artifactDir = await writeRunArtifacts(root, "openrouter", {
      manifest: {
        runId: "openrouter",
        provider: "openrouter",
        authMode: "api",
        query: "test",
        timestamp: new Date().toISOString(),
        status: "success",
      },
      request: {
        present: true,
        url: "https://example.com",
        method: "POST",
        headers: { Authorization: "Bearer test-token" },
        bodyText: '{"q":"hello"}',
        bodyJson: { q: "hello" },
      },
      response: sentinel("none"),
      responseRawText: "raw body",
      streamEvents: [{ line: "data: hello" }],
      normalizedResult: { textResponse: "ok" },
      error: sentinel("none"),
    });

    await writeIndex(root, [
      {
        runId: "openrouter",
        provider: "openrouter",
        authMode: "api",
        status: "success",
        artifactDir,
      },
    ]);

    const manifest = JSON.parse(await readFile(join(artifactDir, "manifest.json"), "utf8"));
    const index = JSON.parse(await readFile(join(root, "index.json"), "utf8"));

    expect(manifest.status).toBe("success");
    expect(index.runs).toHaveLength(1);
    expect(index.runs[0].runId).toBe("openrouter");
  });

  it("fetch interceptor captures request/response and SSE lines", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (_input: RequestInfo | URL, _init?: RequestInit) => {
      const body = `data: {"type":"response.output_text.delta","delta":"hello"}\n\ndata: [DONE]\n\n`;
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    }) as typeof fetch;

    try {
      const state = { streamEvents: [] as unknown[] };
      const responseText = await withFetchInterceptor(state, async () => {
        const response = await fetch("https://example.test/sse", {
          method: "POST",
          headers: { Authorization: "Bearer token-123456" },
          body: JSON.stringify({ query: "hello" }),
        });
        return response.text();
      });

      expect(responseText).toContain("response.output_text.delta");
      expect((state.request as any).method).toBe("POST");
      expect((state.response as any).status).toBe(200);
      expect(state.streamEvents.length).toBeGreaterThan(0);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("openai sdk interceptor captures request args and return value", async () => {
    const originalPost = OpenAI.prototype.post;
    OpenAI.prototype.post = (function mockPost(this: any, path: string, options: any) {
      if (path === "/responses") {
        return Promise.resolve({ output_text: "ok", output: [], usage: {}, echoed: options?.body, baseURL: this?.baseURL });
      }
      return originalPost.call(this, path, options);
    }) as any;

    try {
      const state: { request?: unknown; response?: unknown } = {};
      await withOpenAISdkInterceptor(state, async () => {
        const client = new OpenAI({ apiKey: "test-key", baseURL: "https://example.test/v1" });
        await (client as any).post("/responses", { body: { model: "gpt-test", input: "hello" } });
      });

      expect((state.request as any).args).toMatchObject({ model: "gpt-test", input: "hello" });
      expect((state.response as any).value.output_text).toBe("ok");
    } finally {
      OpenAI.prototype.post = originalPost;
    }
  });
});
