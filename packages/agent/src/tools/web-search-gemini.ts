import { GoogleGenAI, ThinkingLevel } from "@google/genai";
import type { SearchAuth, WebSearchResponse, WebSearchToolResult, SearchOptions } from "./web-search";

/**
 * Perform a web search using Google Gemini's grounding with Google Search.
 * Note: Domain filtering is not supported by Gemini and is silently ignored.
 */
export async function geminiWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  const ai = new GoogleGenAI({ apiKey: auth.value });

  const response = await ai.models.generateContent({
    model: options?.model ?? "gemini-3-flash-preview",
    contents: query,
    config: {
      tools: [{ googleSearch: {} }],
      thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
      ...(options?.system ? { systemInstruction: options.system } : {}),
    },
  });

  const textResponse = response.text ?? "";
  const results: (WebSearchToolResult | string)[] = [];

  // Extract grounding metadata (search sources)
  const grounding = response.candidates?.[0]?.groundingMetadata;
  if (grounding?.groundingChunks) {
    const citations = grounding.groundingChunks
      .filter((c) => c.web)
      .map((c) => ({
        title: c.web!.title ?? c.web!.uri ?? "",
        url: c.web!.uri ?? "",
      }));
    if (citations.length > 0) {
      results.push({ tool_use_id: "gemini-search", content: citations });
    }
  }

  return {
    query,
    results,
    textResponse,
    usage: {
      input_tokens: response.usageMetadata?.promptTokenCount ?? 0,
      output_tokens: response.usageMetadata?.candidatesTokenCount ?? 0,
      web_search_requests: 1,
    },
  };
}

// Quick test
if (import.meta.main) {
  const key = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY;
  if (!key) {
    console.error("Set GOOGLE_API_KEY or GEMINI_API_KEY to test Gemini web search.");
    process.exit(1);
  }
  const result = await geminiWebSearch(
    "What is the current price of Bitcoin?",
    { type: "api-key", value: key },
  );
  console.log(JSON.stringify(result, null, 2));
}
