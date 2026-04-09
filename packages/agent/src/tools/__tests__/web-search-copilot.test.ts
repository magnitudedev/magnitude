import { describe, expect, mock, test } from "bun:test";
import { Effect, Layer } from "effect";
import { ProviderAuth, ProviderState } from "@magnitudedev/providers";
import {
  buildCopilotWebSearchRequest,
  copilotWebSearch,
  countCopilotSearchCalls,
  extractCopilotCitations,
  extractCopilotTextResponse,
} from "../web-search-copilot";
import { detectSearchProvider, resolveCopilotAuth, webSearch } from "../web-search";

const originalFetch = globalThis.fetch;
const originalMagnitudeSearchProvider = process.env.MAGNITUDE_SEARCH_PROVIDER;

function createFetchMock(payload?: any) {
  return mock(async (_url: string, _init?: RequestInit) => ({
    ok: true,
    json: async () =>
      payload ?? {
        output_text: "copilot answer",
        output: [],
        usage: { input_tokens: 1, output_tokens: 2 },
      },
    text: async () => "",
  })) as typeof fetch;
}

async function withFetchMock<T>(fetchImpl: typeof fetch, fn: () => Promise<T>): Promise<T> {
  globalThis.fetch = fetchImpl;
  try {
    return await fn();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

const makeProviderStateLayer = (
  providerId: string | null,
  modelId = "claude-sonnet-4.6",
  seenSlots?: string[],
) =>
  Layer.succeed(
    ProviderState,
    {
      peek: (slot: string) => {
        seenSlots?.push(slot);
        return Effect.succeed(
          providerId
            ? {
                model: { providerId, id: modelId } as any,
                auth: null,
              }
            : null,
        );
      },
      getSlot: () => Effect.die("unused"),
      setSelection: () => Effect.die("unused"),
      clear: () => Effect.die("unused"),
      contextWindow: () => Effect.die("unused"),
      contextLimits: () => Effect.die("unused"),
      accumulateUsage: () => Effect.die("unused"),
      getUsage: () => Effect.die("unused"),
      resetUsage: () => Effect.die("unused"),
    },
  );

const makeProviderAuthLayer = (entries: Record<string, any>) =>
  Layer.succeed(
    ProviderAuth,
    {
      loadAuth: () => Effect.succeed(entries),
      getAuth: (providerId: string) => Effect.succeed(entries[providerId]),
      setAuth: () => Effect.die("unused"),
      removeAuth: () => Effect.die("unused"),
      refresh: () => Effect.die("unused"),
      detectProviders: () => Effect.die("unused"),
      detectDefaultProvider: () => Effect.die("unused"),
      detectProviderAuthMethods: () => Effect.die("unused"),
      connectedProviderIds: () => Effect.succeed(new Set(Object.keys(entries))),
    },
  );

describe("web-search-copilot request and normalization", () => {
  test("builds a Copilot responses request with fixed gpt-5.4 model and allowed domains", async () => {
    expect(
      buildCopilotWebSearchRequest("latest AI news", {
        system: "be concise",
        allowed_domains: ["github.com"],
      }),
    ).toEqual({
      model: "gpt-5.4",
      input: "latest AI news",
      tools: [{ type: "web_search", filters: { allowed_domains: ["github.com"] } }],
      include: ["web_search_call.action.sources"],
      instructions: "be concise",
      stream: false,
    });
  });

  test("ignores options.model and still uses the fixed Copilot search model", async () => {
    expect(
      buildCopilotWebSearchRequest("latest AI news", {
        system: "be concise",
        allowed_domains: ["github.com"],
        model: "claude-sonnet-4.6",
      }),
    ).toEqual({
      model: "gpt-5.4",
      input: "latest AI news",
      tools: [{ type: "web_search", filters: { allowed_domains: ["github.com"] } }],
      include: ["web_search_call.action.sources"],
      instructions: "be concise",
      stream: false,
    });
  });

  test("prefers url_citation annotations and deduplicates urls", async () => {
    expect(
      extractCopilotCitations([
        {
          type: "message",
          content: [
            {
              type: "output_text",
              annotations: [
                { type: "url_citation", title: "One", url: "https://example.com/1" },
                { type: "url_citation", title: "Duplicate", url: "https://example.com/1" },
                { type: "url_citation", title: "Two", url: "https://example.com/2" },
              ],
            },
          ],
        },
        {
          type: "web_search_call",
          action: { sources: [{ title: "Ignored", url: "https://ignored.example.com" }] },
        },
      ]),
    ).toEqual([
      {
        tool_use_id: "copilot-search",
        content: [
          { title: "One", url: "https://example.com/1" },
          { title: "Two", url: "https://example.com/2" },
        ],
      },
    ]);
  });

  test("falls back to web_search_call sources when annotations are absent", async () => {
    expect(
      extractCopilotCitations([
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
      ]),
    ).toEqual([
      {
        tool_use_id: "copilot-search",
        content: [
          { title: "Alpha", url: "https://alpha.test" },
          { title: "https://beta.test", url: "https://beta.test" },
        ],
      },
    ]);
  });

  test("normalizes a successful Copilot response", async () => {
    const fetchMock = createFetchMock({
      output: [
        {
          type: "message",
          content: [
            {
              type: "output_text",
              text: "First paragraph.",
              annotations: [{ type: "url_citation", title: "Docs", url: "https://docs.example.com" }],
            },
            { type: "output_text", text: "Second paragraph." },
          ],
        },
        { type: "web_search_call", action: { sources: [{ url: "https://ignored.example.com" }] } },
      ],
      usage: { input_tokens: 11, output_tokens: 29 },
    });

    await withFetchMock(fetchMock, async () => {
      const result = await copilotWebSearch("query", { type: "oauth-token", value: "copilot-token" }, { model: "claude-sonnet-4.6" });

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

      const [url, init] = fetchMock.mock.calls[0]!;
      expect(url).toBe("https://api.githubcopilot.com/responses");
      expect(init?.headers).toMatchObject({
        "Content-Type": "application/json",
        "Authorization": "Bearer copilot-token",
        "Openai-Intent": "conversation-edits",
        "x-initiator": "user",
        "User-Agent": "GitHubCopilotChat/0.35.0",
        "Editor-Version": "vscode/1.107.0",
        "Editor-Plugin-Version": "copilot-chat/0.35.0",
        "Copilot-Integration-Id": "vscode-chat",
      });
      expect(JSON.parse(String(init?.body))).toMatchObject({
        model: "gpt-5.4",
        input: "query",
      });
    });
  });

  test("counts web_search_call items and extracts fallback text", async () => {
    expect(countCopilotSearchCalls([{ type: "message" }, { type: "web_search_call" }, { type: "web_search_call" }])).toBe(2);
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

  test("retries once on 401 and succeeds with refreshed auth", async () => {
    const fetchMock = mock(async (_url: string, init?: RequestInit) => {
      const auth = String((init?.headers as Record<string, string>)?.Authorization ?? "");
      if (auth === "Bearer stale-token") {
        return {
          ok: false,
          status: 401,
          text: async () => "IDE token expired: unauthorized: token expired",
          json: async () => ({}),
        } as any;
      }
      return {
        ok: true,
        json: async () => ({
          output: [
            {
              type: "message",
              content: [
                {
                  type: "output_text",
                  text: "refreshed result",
                  annotations: [{ type: "url_citation", title: "Docs", url: "https://docs.example.com" }],
                },
              ],
            },
            { type: "web_search_call", action: { sources: [{ url: "https://docs.example.com" }] } },
          ],
          usage: { input_tokens: 1, output_tokens: 2 },
        }),
        text: async () => "",
      } as any;
    }) as typeof fetch;

    await withFetchMock(fetchMock, async () => {
      const resolver = mock(async () => ({ type: "oauth-token" as const, value: "fresh-token" }));
      const result = await copilotWebSearch(
        "query",
        { type: "oauth-token", value: "stale-token" },
        undefined,
        resolver,
      );

      expect(resolver).toHaveBeenCalledTimes(1);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(result.results).toEqual([
        {
          tool_use_id: "copilot-search",
          content: [{ title: "Docs", url: "https://docs.example.com" }],
        },
      ]);
    });
  });

  test("fails clearly when 401 persists after one refresh retry", async () => {
    const fetchMock = mock(async () => ({
      ok: false,
      status: 401,
      text: async () => "unauthorized: token expired",
      json: async () => ({}),
    })) as typeof fetch;

    await withFetchMock(fetchMock, async () => {
      await expect(
        copilotWebSearch(
          "query",
          { type: "oauth-token", value: "stale-token" },
          undefined,
          async () => ({ type: "oauth-token", value: "still-bad-token" }),
        ),
      ).rejects.toThrow("GitHub Copilot web search unauthorized (401) after auth refresh retry.");
    });
  });

  test("retries once when first attempt times out and then succeeds", async () => {
    let calls = 0;
    const fetchMock = mock(async () => {
      calls += 1;
      if (calls === 1) {
        const error = new Error("aborted");
        (error as Error & { name: string }).name = "AbortError";
        throw error;
      }

      return {
        ok: true,
        json: async () => ({
          output: [
            {
              type: "message",
              content: [
                {
                  type: "output_text",
                  text: "retry success",
                  annotations: [{ type: "url_citation", title: "Docs", url: "https://docs.example.com" }],
                },
              ],
            },
            { type: "web_search_call", action: { sources: [{ url: "https://docs.example.com" }] } },
          ],
          usage: { input_tokens: 1, output_tokens: 2 },
        }),
        text: async () => "",
      } as any;
    }) as typeof fetch;

    await withFetchMock(fetchMock, async () => {
      const result = await copilotWebSearch("query", { type: "oauth-token", value: "copilot-token" });
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(result.results).toEqual([
        {
          tool_use_id: "copilot-search",
          content: [{ title: "Docs", url: "https://docs.example.com" }],
        },
      ]);
      expect(result.usage.web_search_requests).toBe(1);
    });
  });

  test("fails clearly after timeout retry is exhausted", async () => {
    const abortingFetch = mock(async () => {
      const error = new Error("aborted");
      (error as Error & { name: string }).name = "AbortError";
      throw error;
    }) as typeof fetch;

    await withFetchMock(abortingFetch, async () => {
      await expect(
        copilotWebSearch("query", { type: "oauth-token", value: "copilot-token" }),
      ).rejects.toThrow("GitHub Copilot web search timed out after 90000ms");
      expect(abortingFetch).toHaveBeenCalledTimes(2);
    });
  });
});


describe("web-search Copilot routing", () => {
  test("accepts MAGNITUDE_SEARCH_PROVIDER=github-copilot", async () => {
    process.env.MAGNITUDE_SEARCH_PROVIDER = "github-copilot";
    try {
      expect(detectSearchProvider("openai")).toBe("github-copilot");
    } finally {
      process.env.MAGNITUDE_SEARCH_PROVIDER = originalMagnitudeSearchProvider;
    }
  });

  test("providerId github-copilot selects the Copilot backend", async () => {
    delete process.env.MAGNITUDE_SEARCH_PROVIDER;
    expect(detectSearchProvider("github-copilot")).toBe("github-copilot");
    process.env.MAGNITUDE_SEARCH_PROVIDER = originalMagnitudeSearchProvider;
  });

  test("invalid override mentions github-copilot among valid values", async () => {
    process.env.MAGNITUDE_SEARCH_PROVIDER = "bogus";
    try {
      expect(() => detectSearchProvider("openai")).toThrow(
        'Invalid MAGNITUDE_SEARCH_PROVIDER value "bogus". Must be one of: anthropic, openai, gemini, openrouter, vercel, github-copilot.',
      );
    } finally {
      process.env.MAGNITUDE_SEARCH_PROVIDER = originalMagnitudeSearchProvider;
    }
  });

  test("resolveCopilotAuth only accepts OAuth auth", async () => {
    expect(resolveCopilotAuth({ type: "oauth", accessToken: "copilot-token" } as any)).toEqual({
      type: "oauth-token",
      value: "copilot-token",
    });
    expect(() => resolveCopilotAuth()).toThrow(
      "No GitHub Copilot OAuth session available for web search. Authenticate GitHub Copilot in the app.",
    );
  });

  test("webSearch uses the lead slot and ignores the selected Copilot slot model", async () => {
    const fetchMock = createFetchMock();
    await withFetchMock(fetchMock, async () => {
      const seenSlots: string[] = [];
      const result = await Effect.runPromise(
        webSearch("query").pipe(
          Effect.provide(
            Layer.mergeAll(
              makeProviderStateLayer("github-copilot", "claude-sonnet-4.6", seenSlots),
              makeProviderAuthLayer({
                "github-copilot": { type: "oauth", accessToken: "copilot-token" },
              }),
            ),
          ),
        ),
      );

      expect(result.query).toBe("query");
      expect(seenSlots).toEqual(["lead"]);

      const [, init] = fetchMock.mock.calls[0]!;
      const body = JSON.parse(String(init?.body));
      expect(body.model).toBe("gpt-5.4");
    });
  });

  test("webSearch pre-refreshes Copilot auth before first request when refresh token exists", async () => {
    const fetchMock = createFetchMock();
    const refresh = mock(async () => ({ type: "oauth", accessToken: "fresh-token", refreshToken: "refresh-2" } as any));
    const setAuth = mock(async () => undefined);

    await withFetchMock(fetchMock, async () => {
      await Effect.runPromise(
        webSearch("query").pipe(
          Effect.provide(
            Layer.mergeAll(
              makeProviderStateLayer("github-copilot", "claude-sonnet-4.6"),
              Layer.succeed(
                ProviderAuth,
                {
                  loadAuth: () => Effect.succeed({}),
                  getAuth: () =>
                    Effect.succeed({
                      type: "oauth",
                      accessToken: "stale-token",
                      refreshToken: "refresh-1",
                    } as any),
                  setAuth: (_providerId: string, auth: any) => Effect.promise(() => setAuth(auth)),
                  removeAuth: () => Effect.die("unused"),
                  refresh: (_providerId: string, refreshToken: string) =>
                    Effect.promise(() => refresh(refreshToken)),
                  detectProviders: () => Effect.die("unused"),
                  detectDefaultProvider: () => Effect.die("unused"),
                  detectProviderAuthMethods: () => Effect.die("unused"),
                  connectedProviderIds: () => Effect.succeed(new Set(["github-copilot"])),
                },
              ),
            ),
          ),
        ),
      );

      expect(refresh).toHaveBeenCalledTimes(1);
      expect(setAuth).toHaveBeenCalledTimes(1);

      const [, init] = fetchMock.mock.calls[0]!;
      expect((init?.headers as Record<string, string>).Authorization).toBe("Bearer fresh-token");
    });
  });

  test("webSearch ignores options.model when routed through GitHub Copilot", async () => {
    const fetchMock = createFetchMock();
    await withFetchMock(fetchMock, async () => {
      await Effect.runPromise(
        webSearch("query", { model: "gpt-5-mini" }).pipe(
          Effect.provide(
            Layer.mergeAll(
              makeProviderStateLayer("github-copilot", "gpt-5-mini"),
              makeProviderAuthLayer({
                "github-copilot": { type: "oauth", accessToken: "copilot-token" },
              }),
            ),
          ),
        ),
      );

      const [, init] = fetchMock.mock.calls[0]!;
      const body = JSON.parse(String(init?.body));
      expect(body.model).toBe("gpt-5.4");
    });
  });

  test("MAGNITUDE_SEARCH_PROVIDER=github-copilot still uses fixed gpt-5.4 regardless of slot model", async () => {
    const fetchMock = createFetchMock();
    process.env.MAGNITUDE_SEARCH_PROVIDER = "github-copilot";

    try {
      await withFetchMock(fetchMock, async () => {
        await Effect.runPromise(
          webSearch("query", { model: "claude-sonnet-4.6" }).pipe(
            Effect.provide(
              Layer.mergeAll(
                makeProviderStateLayer("openai", "gpt-5"),
                makeProviderAuthLayer({
                  "github-copilot": { type: "oauth", accessToken: "copilot-token" },
                }),
              ),
            ),
          ),
        );

        const [, init] = fetchMock.mock.calls[0]!;
        const body = JSON.parse(String(init?.body));
        expect(body.model).toBe("gpt-5.4");
      });
    } finally {
      process.env.MAGNITUDE_SEARCH_PROVIDER = originalMagnitudeSearchProvider;
    }
  });
});
