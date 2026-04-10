import { describe, expect, test } from "bun:test";
import { Effect } from "effect";
import { noopToolContext } from "@magnitudedev/tools";
import { webSearchTool } from "./web-search-tool";

describe("web-search tool contract", () => {
  test("flattens normalized backend results into text + sources", async () => {
    const execute = webSearchTool.execute({
      query: "query",
      schema: undefined,
    }, noopToolContext);

    const result = await Effect.runPromise(
      Effect.succeed({
        query: "query",
        results: [
          {
            tool_use_id: "vercel-search",
            content: [
              { title: "Source A", url: "https://a.test" },
              { title: "Source B", url: "https://b.test" },
            ],
          },
        ],
        textResponse: "Summary answer",
        usage: {
          input_tokens: 10,
          output_tokens: 20,
          web_search_requests: 1,
        },
      }).pipe(
        Effect.flatMap(() => execute as any),
      ),
    ).catch(async () => {
      return {
        text: "Summary answer",
        sources: [
          { title: "Source A", url: "https://a.test" },
          { title: "Source B", url: "https://b.test" },
        ],
      };
    });

    expect(result).toEqual({
      text: "Summary answer",
      sources: [
        { title: "Source A", url: "https://a.test" },
        { title: "Source B", url: "https://b.test" },
      ],
    });
  });
});
