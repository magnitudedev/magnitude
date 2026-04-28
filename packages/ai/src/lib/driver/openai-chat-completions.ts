import { Effect, Schema, Stream } from "effect"
import type { Driver } from "./driver"
import { sseChunks } from "./sse"
import {
  AuthFailed,
  InvalidRequest,
  ParseError,
  RateLimited,
  TransportError,
  UsageLimitExceeded,
} from "../errors/model-error"
import { ChatCompletionsStreamChunk } from "../wire/chat-completions"

function classifyHttpError(
  providerId: string,
  status: number,
  body: string,
) {
  if (status === 401 || status === 403) {
    return new AuthFailed({
      providerId,
      status,
      message: body || `Authentication failed with status ${status}`,
    })
  }

  if (status === 429) {
    return new RateLimited({
      providerId,
      status,
      message: body || "Rate limited",
      retryAfterMs: null,
    })
  }

  if (status === 402) {
    return new UsageLimitExceeded({
      providerId,
      status,
      message: body || "Usage limit exceeded",
    })
  }

  if (status === 400 || status === 404 || status === 422) {
    return new InvalidRequest({
      providerId,
      status,
      message: body || `Invalid request with status ${status}`,
    })
  }

  return new TransportError({
    providerId,
    status,
    message: body || `HTTP ${status}`,
    retryable: status >= 500,
  })
}

function joinUrl(endpoint: string, path: string): string {
  return `${endpoint.replace(/\/+$/, "")}${path}`
}

async function* responseBodyIterator(
  providerId: string,
  response: Response,
): AsyncGenerator<Uint8Array, void, void> {
  const body = response.body
  if (!body) {
    throw new TransportError({
      providerId,
      status: response.status,
      message: "Response body stream missing",
      retryable: false,
    })
  }

  const reader = body.getReader()
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        return
      }
      if (value) {
        yield value
      }
    }
  } catch (cause) {
    throw new TransportError({
      providerId,
      status: response.status,
      message: `Failed to read response stream: ${String(cause)}`,
      retryable: true,
    })
  } finally {
    void reader.cancel()
  }
}

export const openAIChatCompletionsDriver: Driver = {
  id: "openai-chat-completions",
  stream: (request, endpoint, authToken) =>
    Stream.unwrap(
      Effect.tryPromise({
        try: async () => {
          const providerId = endpoint
          const response = await fetch(joinUrl(endpoint, "/chat/completions"), {
            method: "POST",
            headers: {
              Authorization: `Bearer ${authToken}`,
              Accept: "text/event-stream",
              "Content-Type": "application/json",
            },
            body: JSON.stringify(request),
          })

          if (!response.ok) {
            const body = await response.text().catch(() => "")
            throw classifyHttpError(providerId, response.status, body)
          }

          const byteStream = Stream.fromAsyncIterable(
            responseBodyIterator(providerId, response),
            (cause) =>
              cause instanceof TransportError
                ? cause
                : new TransportError({
                    providerId,
                    status: response.status,
                    message: `Failed to read response stream: ${String(cause)}`,
                    retryable: true,
                  }),
          )

          return sseChunks(providerId, byteStream).pipe(
            Stream.mapEffect((json) =>
              Schema.decodeUnknown(ChatCompletionsStreamChunk)(json).pipe(
                Effect.mapError(
                  (cause) =>
                    new ParseError({
                      providerId,
                      message: `Failed to decode chat completion chunk: ${String(cause)}`,
                    }),
                ),
              ),
            ),
          )
        },
        catch: (cause) =>
          cause instanceof AuthFailed ||
          cause instanceof InvalidRequest ||
          cause instanceof ParseError ||
          cause instanceof RateLimited ||
          cause instanceof TransportError ||
          cause instanceof UsageLimitExceeded
            ? cause
            : new TransportError({
                providerId: endpoint,
                status: null,
                message: `HTTP request failed: ${String(cause)}`,
                retryable: true,
              }),
      }),
    ),
}
