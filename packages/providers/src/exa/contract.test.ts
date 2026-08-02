import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  ExaSearchRequestSchema,
  ExaSearchResponseSchema,
} from "./contract"

describe("Exa search contract", () => {
  it("round-trips the complete documented JSON request shape", () => {
    const wire = {
      query: "latest developments in LLMs",
      type: "deep",
      stream: false,
      numResults: 10,
      category: "publication",
      userLocation: "US",
      includeDomains: ["arxiv.org"],
      excludeDomains: ["example.com"],
      startCrawlDate: "2026-01-01T00:00:00.000Z",
      endCrawlDate: "2026-07-01T00:00:00.000Z",
      startPublishedDate: "2026-01-01T00:00:00.000Z",
      endPublishedDate: "2026-07-01T00:00:00.000Z",
      includeText: ["language models"],
      excludeText: ["advertisement"],
      flags: ["experimental-ranking"],
      moderation: true,
      useAutoprompt: false,
      additionalQueries: ["LLM research", "language model research"],
      contents: {
        text: {
          maxCharacters: 10_000,
          includeHtmlTags: false,
          verbosity: "standard",
          includeSections: ["body", "metadata"],
          excludeSections: ["navigation", "footer"],
        },
        highlights: {
          query: "important findings",
          maxCharacters: 2_000,
          numSentences: 3,
          highlightsPerUrl: 2,
        },
        summary: {
          query: "summarize the findings",
          schema: {
            type: "object",
            properties: {
              summary: { type: "string" },
            },
          },
        },
        livecrawl: "fallback",
        context: { maxCharacters: 4_000 },
        livecrawlTimeout: 10_000,
        maxAgeHours: 24,
        filterEmptyResults: true,
        subpages: 2,
        subpageTarget: ["research", "methods"],
        extras: {
          links: 5,
          imageLinks: 3,
        },
      },
      outputSchema: {
        type: "object",
        description: "Extract the main finding",
        properties: {
          finding: { type: "string" },
        },
        required: ["finding"],
        additionalProperties: false,
      },
      systemPrompt: "Prefer primary sources.",
      compliance: "hipaa",
    } as const

    const decoded = Schema.decodeUnknownSync(ExaSearchRequestSchema)(wire)
    expect(Schema.encodeSync(ExaSearchRequestSchema)(decoded)).toEqual(wire)
  })

  it("decodes the complete current JSON response shape", () => {
    const wire = {
      requestId: "request-1",
      results: [{
        id: "https://magnitude.dev/",
        title: "Magnitude",
        url: "https://magnitude.dev/",
        publishedDate: null,
        author: "Magnitude",
        score: 0.92,
        image: "https://magnitude.dev/og-image.png",
        favicon: "https://magnitude.dev/favicon.ico",
        text: "Magnitude is a coding agent.",
        highlights: ["The best coding agent for open models."],
        highlightScores: [0.84],
        summary: "An open-model coding agent.",
        subpages: [{
          id: "https://magnitude.dev/docs",
          title: null,
          url: "https://magnitude.dev/docs",
          author: null,
          highlights: ["Documentation"],
          extras: {
            links: ["https://docs.magnitude.dev"],
            imageLinks: ["https://magnitude.dev/og-image.png"],
          },
        }],
        extras: {
          links: ["https://docs.magnitude.dev"],
          imageLinks: ["https://magnitude.dev/og-image.png"],
        },
        entities: [{
          id: "https://exa.ai/library/organization/magnitude",
          type: "company",
          version: 1,
          properties: {
            name: "Magnitude",
            foundedYear: 2025,
            description: "Open-model coding agent",
            workforce: { total: 2 },
            headquarters: {
              address: "San Francisco, CA",
              city: "San Francisco",
              postalCode: null,
              country: "United States",
            },
            financials: {
              revenueAnnual: null,
              fundingTotal: 1_000_000,
              fundingLatestRound: {
                name: "Seed",
                date: "2025-01",
                amount: 1_000_000,
              },
            },
            webTraffic: {
              visitsMonthly: 680,
              countryRank: null,
              avgDurationSeconds: 0,
              history: [{
                value: 680,
                dateFrom: "2026-04",
                dateTo: "2026-04",
              }],
            },
            research: {
              worksCount: 1,
              citationCount: 2,
              areas: ["coding agents"],
              notableWorks: [{
                title: "Magnitude",
                year: 2026,
                venue: "GitHub",
                citationCount: 2,
                doi: null,
                id: "magnitude",
              }],
              topResearchers: [{
                person: {
                  name: "Researcher",
                  id: "researcher-1",
                },
                worksCount: 1,
                citationCount: 2,
              }],
            },
          },
        }, {
          id: "https://exa.ai/library/person/researcher",
          type: "person",
          version: 1,
          properties: {
            name: "Researcher",
            location: "San Francisco",
            firstName: "Test",
            lastName: "Researcher",
            workHistory: [{
              title: "Engineer",
              location: "San Francisco",
              dates: {
                from: "2025-01",
                to: null,
              },
              company: {
                id: "magnitude",
                name: "Magnitude",
              },
            }],
            educationHistory: [{
              degree: "PhD, Computer Science",
              dates: null,
              institution: {
                id: null,
                name: "Example University",
              },
            }],
            research: null,
          },
        }],
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
      statuses: [{
        id: "https://magnitude.dev/",
        status: "success",
      }],
      costDollars: {
        total: 0.007,
        search: {
          neural: 0.007,
          keyword: 0,
        },
        contents: {
          text: 0,
          highlights: 0,
          summary: 0,
        },
      },
      resolvedSearchType: "",
      searchTime: 998.3,
      context: "Deprecated combined context",
      autoDate: "2026-07-30",
    } as const

    expect(Schema.decodeUnknownSync(ExaSearchResponseSchema)(wire)).toBeDefined()
  })

  it("rejects the old flattened structured-output mock", () => {
    expect(() => Schema.decodeUnknownSync(ExaSearchResponseSchema)({
      requestId: "request-1",
      results: [],
      output: { answer: 42 },
    })).toThrow()
  })
})
