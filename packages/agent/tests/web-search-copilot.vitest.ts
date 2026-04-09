import { describe, expect, it, vi } from "vitest";
import { Effect, Layer } from "effect";
import { ProviderAuth, ProviderState } from "@magnitudedev/providers";
import {
  buildCopilotWebSearchRequest,
  copilotWebSearch,
} from "../src/tools/web-search-copilot";
import { detectSearchProvider, webSearch } from "../src/tools/web-search";

function makeProviderState(providerId: string | null, modelId?: string) {
  return Layer.succeed(ProviderState, {
    peek: (_slot: string) =>
      Effect.succeed(providerId ? { model: { providerId, id: modelId } } : null),
  } as any);
}

function makeProviderAuth(authByProvider: Record<string, any | undefined>) {
  return Layer.succeed(ProviderAuth, {
    getAuth: (providerId: string) => Effect.succeed(authByProvider[providerId]),
  } as any);
}

async function withFetchMock<T>(fetchImpl: typeof fetch, run: () => Promise<T>): Promise<T> {
  const originalFetch = globalThis.fetch;
  (globalThis as any).fetch = fetchImpl;
  try {
    return await run();
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

describe("github-copilot web search backend/router", () => {
  it("builds Copilot direct responses request with web_search tool + include", () => {
    expect(buildCopilotWebSearchRequest("latest news", {
      allowed_domains: ["github.com"],
      system: "be concise",
    })).toEqual({
      model: "gpt-5.4",
      input: "latest news",
      tools: [{ type: "web_search", filters: { allowed_domains: ["github.com"] } }],
      include: ["web_search_call.action.sources"],
      instructions: "be concise",
    });
  });

  it("copilotWebSearch posts to api.githubcopilot.com/responses and normalizes citations", async () => {
    let capturedUrl = "";
    let capturedInit: RequestInit | undefined;

    await withFetchMock(
      vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
        capturedUrl = String(url);
        capturedInit = init;
        return new Response(JSON.stringify({
          output_text: "Copilot answer",
          output: [
            {
              type: "message",
              content: [
                {
                  type: "output_text",
                  text: "Copilot answer",
                  annotations: [{ type: "url_citation", title: "Docs", url: "https://docs.example.com" }],
                },
              ],
            },
            {
              type: "web_search_call",
              action: { sources: [{ url: "https://docs.example.com" }, { url: "https://extra.example.com" }] },
            },
          ],
          usage: { input_tokens: 3, output_tokens: 5 },
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }) as any,
      async () => {
        const result = await copilotWebSearch("query", { type: "oauth-token", value: "copilot-token" });
        expect(result.results).toEqual([
          {
            tool_use_id: "copilot-search",
            content: [
              { title: "Docs", url: "https://docs.example.com" },
              { title: "https://extra.example.com", url: "https://extra.example.com" },
            ],
          },
        ]);
      },
    );

    expect(capturedUrl).toBe("https://api.githubcopilot.com/responses");
    expect(capturedInit?.headers).toMatchObject({
      Authorization: "Bearer copilot-token",
      "Openai-Intent": "conversation-edits",
      "x-initiator": "user",
    });
  });

  it("detectSearchProvider recognizes github-copilot", () => {
    delete process.env.MAGNITUDE_SEARCH_PROVIDER;
    expect(detectSearchProvider("github-copilot")).toBe("github-copilot");
  });

  it("webSearch routes github-copilot with stored OAuth auth", async () => {
    let capturedUrl = "";

    await withFetchMock(
      vi.fn(async (url: string | URL | Request) => {
        capturedUrl = String(url);
        return new Response(JSON.stringify({
          output_text: "Copilot answer",
          output: [],
          usage: { input_tokens: 1, output_tokens: 2, web_search_requests: 1 },
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }) as any,
      async () => {
        const result = await Effect.runPromise(
          webSearch("query").pipe(
            Effect.provide(
              Layer.mergeAll(
                makeProviderState("github-copilot", "gpt-5.4"),
                makeProviderAuth({ "github-copilot": { type: "oauth", accessToken: "copilot-token" } }),
              ),
            ),
          ) as any,
        );
        expect(result.textResponse).toBe("Copilot answer");
      },
    );

    expect(capturedUrl).toBe("https://api.githubcopilot.com/responses");
  });

  it("webSearch missing github-copilot OAuth auth errors clearly", async () => {
    await expect(
      Effect.runPromise(
        webSearch("query").pipe(
          Effect.provide(
            Layer.mergeAll(
              makeProviderState("github-copilot", "gpt-5.4"),
              makeProviderAuth({}),
            ),
          ),
        ) as any,
      ),
    ).rejects.toThrow(
      "No GitHub Copilot OAuth session available for web search. Authenticate GitHub Copilot in the app.",
    );
  });
});
