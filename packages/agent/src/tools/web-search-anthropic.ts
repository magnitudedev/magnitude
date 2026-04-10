import Anthropic from "@anthropic-ai/sdk";
import type { SearchAuth, WebSearchResponse, WebSearchToolResult, SearchOptions } from "./web-search";

type CreateArgs = Parameters<Anthropic["beta"]["messages"]["create"]>[0];
type CreateResult = Awaited<ReturnType<Anthropic["beta"]["messages"]["create"]>>;
type StreamArgs = Parameters<Anthropic["beta"]["messages"]["stream"]>[0];
type StreamResult = Awaited<ReturnType<Anthropic["beta"]["messages"]["stream"]>>;
type FinalMessageResult = Awaited<ReturnType<StreamResult["finalMessage"]>>;

type AnthropicInterceptor = {
  onCreate?: (
    payload: { client: Record<string, unknown>; args: CreateArgs },
    next: () => Promise<CreateResult>,
  ) => Promise<CreateResult>;
  onStream?: (
    payload: { client: Record<string, unknown>; args: StreamArgs },
    next: () => Promise<StreamResult>,
  ) => Promise<StreamResult>;
};

let anthropicInterceptor: AnthropicInterceptor | null = null;

function createAnthropicClient(auth: SearchAuth): Anthropic {
  return auth.type === "oauth-token"
    ? new Anthropic({ authToken: auth.value })
    : new Anthropic({ apiKey: auth.value });
}

async function runCreate(client: Anthropic, args: CreateArgs): Promise<CreateResult> {
  const next = () => (client.beta.messages.create as Function)(args) as Promise<CreateResult>;
  if (!anthropicInterceptor?.onCreate) return next();
  return anthropicInterceptor.onCreate({
    client: { provider: "anthropic" },
    args,
  }, next);
}

async function runStream(client: Anthropic, args: StreamArgs): Promise<StreamResult> {
  const next = () => (client.beta.messages.stream as Function)(args) as Promise<StreamResult>;
  if (!anthropicInterceptor?.onStream) return next();
  return anthropicInterceptor.onStream({
    client: { provider: "anthropic" },
    args,
  }, next);
}

/**
 * Perform a web search using Anthropic's API with web_search tool.
 */
export async function anthropicWebSearch(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): Promise<WebSearchResponse> {
  const client = createAnthropicClient(auth);

  const model = options?.model ?? "claude-haiku-4-5";
  const max_tokens = options?.max_tokens ?? 4096;

  const response = await runCreate(client, {
    model,
    max_tokens,
    betas: ["web-search-2025-03-05"],
    ...(options?.system && { system: options.system }),
    messages: [{ role: "user", content: query }],
    tools: [
      {
        type: "web_search_20250305",
        name: "web_search",
        ...(options?.allowed_domains && {
          allowed_domains: options.allowed_domains,
        }),
        ...(options?.blocked_domains && {
          blocked_domains: options.blocked_domains,
        }),
        max_uses: 8,
      },
    ],
  });

  const results: (WebSearchToolResult | string)[] = [];
  let textResponse = "";

  for (const block of response.content) {
    if (block.type === "text") {
      textResponse += block.text;
    } else if (block.type === "web_search_tool_result") {
      if (Array.isArray(block.content)) {
        results.push({
          tool_use_id: block.tool_use_id,
          content: block.content.map((r: { title: string; url: string }) => ({
            title: r.title,
            url: r.url,
          })),
        });
      } else if (block.content?.type === "web_search_tool_result_error") {
        results.push(`Web search error: ${block.content.error_code}`);
      }
    }
  }

  return {
    query,
    results,
    textResponse,
    usage: {
      input_tokens: response.usage.input_tokens,
      output_tokens: response.usage.output_tokens,
      web_search_requests:
        response.usage.server_tool_use?.web_search_requests ?? 0,
    },
  };
}

/**
 * Streaming version - yields search progress and results (Anthropic-only).
 */
export async function* anthropicWebSearchStream(
  query: string,
  auth: SearchAuth,
  options?: SearchOptions,
): AsyncGenerator<
  | { type: "search_started"; query: string }
  | { type: "search_result"; result: WebSearchToolResult }
  | { type: "text_delta"; text: string }
  | { type: "done"; response: WebSearchResponse }
> {
  const client =
    auth.type === "oauth-token"
      ? new Anthropic({ authToken: auth.value })
      : new Anthropic({ apiKey: auth.value });

  const model = options?.model ?? "claude-haiku-4-5";
  const max_tokens = options?.max_tokens ?? 4096;

  const stream = await runStream(client, {
    model,
    max_tokens,
    betas: ["web-search-2025-03-05"],
    messages: [{ role: "user", content: query }],
    tools: [
      {
        type: "web_search_20250305",
        name: "web_search",
        ...(options?.allowed_domains && {
          allowed_domains: options.allowed_domains,
        }),
        ...(options?.blocked_domains && {
          blocked_domains: options.blocked_domains,
        }),
        max_uses: 8,
      },
    ],
  });

  const results: (WebSearchToolResult | string)[] = [];
  let textResponse = "";
  let currentSearchQuery = "";

  for await (const event of stream) {
    if (event.type === "content_block_start") {
      const block = event.content_block;
      if (block.type === "server_tool_use") {
        currentSearchQuery = "";
      } else if (block.type === "web_search_tool_result") {
        if (Array.isArray(block.content)) {
          const result: WebSearchToolResult = {
            tool_use_id: block.tool_use_id,
            content: block.content.map((r: { title: string; url: string }) => ({
              title: r.title,
              url: r.url,
            })),
          };
          results.push(result);
          yield { type: "search_result", result };
        } else if (block.content?.type === "web_search_tool_result_error") {
          results.push(`Web search error: ${block.content.error_code}`);
        }
      }
    } else if (event.type === "content_block_delta") {
      const delta = event.delta;
      if (delta.type === "input_json_delta" && delta.partial_json) {
        const match = delta.partial_json.match(/"query"\s*:\s*"([^"]+)"/);
        if (match && match[1] !== currentSearchQuery) {
          currentSearchQuery = match[1];
          yield { type: "search_started", query: currentSearchQuery };
        }
      } else if (delta.type === "text_delta") {
        textResponse += delta.text;
        yield { type: "text_delta", text: delta.text };
      }
    }
  }

  const finalMessage = await stream.finalMessage();

  yield {
    type: "done",
    response: {
      query,
      results,
      textResponse,
      usage: {
        input_tokens: finalMessage.usage.input_tokens,
        output_tokens: finalMessage.usage.output_tokens,
        web_search_requests:
          finalMessage.usage.server_tool_use?.web_search_requests ?? 0,
      },
    },
  };
}

export const __captureOnly = {
  async withInterceptor<T>(interceptor: AnthropicInterceptor, run: () => Promise<T>): Promise<T> {
    const previous = anthropicInterceptor;
    anthropicInterceptor = interceptor;
    try {
      return await run();
    } finally {
      anthropicInterceptor = previous;
    }
  },

  wrapStream(
    stream: StreamResult,
    onEvent: (event: unknown) => void | Promise<void>,
    onFinalMessage?: (message: FinalMessageResult) => void | Promise<void>,
  ): StreamResult {
    return {
      [Symbol.asyncIterator]: async function* () {
        for await (const event of stream as any as AsyncIterable<unknown>) {
          await onEvent(event);
          yield event;
        }
      },
      finalMessage: async () => {
        const finalMessage = await stream.finalMessage();
        if (onFinalMessage) {
          await onFinalMessage(finalMessage);
        }
        return finalMessage;
      },
    } as StreamResult;
  },
};

