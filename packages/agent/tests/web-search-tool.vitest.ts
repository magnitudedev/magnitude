import { FetchHttpClient } from "@effect/platform"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ProviderClient,
  createProviderClient,
} from "@magnitudedev/sdk"
import { webSearchTool } from "../src/tools/web-search"

describe("web search tool", () => {
  it("returns Exa output.content as structured tool data", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({
        requestId: "request-1",
        results: [{
          id: "https://magnitude.dev/",
          title: "Magnitude",
          url: "https://magnitude.dev/",
          highlights: ["The coding agent for open models."],
        }],
        output: {
          content: {
            officialWebsite: "https://magnitude.dev/",
          },
          grounding: [{
            field: "officialWebsite",
            citations: [{
              url: "https://magnitude.dev/",
              title: "Magnitude",
            }],
            confidence: "high",
          }],
        },
        costDollars: {
          total: 0.007,
          search: {
            neural: 0.007,
          },
        },
        resolvedSearchType: "",
        searchTime: 100,
      }),
    })

    try {
      const client = createProviderClient({
        apiKey: " ",
        exaApiKey: "exa_test",
        exaEndpoint: `http://127.0.0.1:${server.port}/search`,
      })
      const result = await Effect.runPromise(
        webSearchTool.execute({
          query: "Magnitude coding agent",
          schema: Option.some({
            type: "object",
            properties: {
              officialWebsite: { type: "string" },
            },
            required: ["officialWebsite"],
            additionalProperties: false,
          }),
        }, { emit: undefined as never }).pipe(
          Effect.provideService(ProviderClient, client),
          Effect.provide(FetchHttpClient.layer),
        ),
      )

      expect(Option.getOrUndefined(result.data)).toEqual({
        officialWebsite: "https://magnitude.dev/",
      })
      expect(result.sources).toEqual([{
        title: "Magnitude",
        url: "https://magnitude.dev/",
      }])
    } finally {
      server.stop(true)
    }
  })
})
