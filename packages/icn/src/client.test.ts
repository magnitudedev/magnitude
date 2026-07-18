import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientResponse from "@effect/platform/HttpClientResponse";
import { Chunk, Effect, Layer, Option, Stream } from "effect";
import { describe, expect, it } from "vitest";
import type { ChatCompletionRequestEncoded } from "./generated/index.js";
import { IcnClientError, makeIcnClient } from "./client.js";

const request: ChatCompletionRequestEncoded = {
  messages: [{ role: "user", content: "hello" }],
  stream: true,
};

const completion = {
  id: "chatcmpl-test",
  object: "chat.completion.chunk",
  created: 1,
  model: "test",
  choices: [
    {
      index: 0,
      delta: { content: "hello" },
    },
  ],
};

const provide = <A, E>(
  http: HttpClient.HttpClient,
  effect: Effect.Effect<A, E, HttpClient.HttpClient>
) => effect.pipe(Effect.provide(Layer.succeed(HttpClient.HttpClient, http)));

describe("IcnClient", () => {
  it("derives ordinary HTTP calls and decodes finite SSE through generated schemas", async () => {
    const seen: string[] = [];
    const http = HttpClient.make((httpRequest) => {
      seen.push(new URL(httpRequest.url).pathname);
      const path = new URL(httpRequest.url).pathname;
      const response =
        path === "/health"
          ? JSON.stringify({ status: "ok", version: "0.1.0" })
          : [
              `data: ${JSON.stringify(completion)}`,
              "",
              "data: [DONE]",
              "",
            ].join("\n");
      return Effect.succeed(
        HttpClientResponse.fromWeb(
          httpRequest,
          new Response(response, {
            status: 200,
            headers: {
              "content-type":
                path === "/health" ? "application/json" : "text/event-stream",
            },
          })
        )
      );
    });

    const result = await Effect.runPromise(
      provide(
        http,
        Effect.gen(function* () {
          const client = yield* makeIcnClient(new URL("http://127.0.0.1:8080"));
          const health = yield* client.health;
          const chunks = yield* Stream.runCollect(
            client.chatCompletion(request)
          );
          return { health, chunks: Chunk.toReadonlyArray(chunks) };
        })
      )
    );

    expect(result.health).toEqual({ status: "ok", version: "0.1.0" });
    expect(result.chunks).toHaveLength(1);
    expect(Option.getOrNull(result.chunks[0]!.choices[0]!.delta.content)).toBe(
      "hello"
    );
    expect(seen).toEqual(["/health", "/v1/chat/completions"]);
  });

  it("fails a completion that reaches EOF without the declared sentinel", async () => {
    const http = HttpClient.make((httpRequest) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          httpRequest,
          new Response(`data: ${JSON.stringify(completion)}\n\n`, {
            status: 200,
            headers: { "content-type": "text/event-stream" },
          })
        )
      )
    );
    const error = await Effect.runPromise(
      provide(
        http,
        Effect.gen(function* () {
          const client = yield* makeIcnClient(new URL("http://127.0.0.1:8080"));
          return yield* Stream.runDrain(client.chatCompletion(request)).pipe(
            Effect.flip
          );
        })
      )
    );
    expect(error).toBeInstanceOf(IcnClientError);
    expect(error.reason).toBe("incomplete-stream");
  });

  it("surfaces structured remote stream errors as typed failures", async () => {
    const errorChunk = {
      ...completion,
      choices: [],
      error: { type: "server_error", code: "overloaded", message: "busy" },
    };
    const http = HttpClient.make((httpRequest) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          httpRequest,
          new Response(
            [
              `data: ${JSON.stringify(errorChunk)}`,
              "",
              "data: [DONE]",
              "",
            ].join("\n"),
            { status: 200, headers: { "content-type": "text/event-stream" } }
          )
        )
      )
    );
    const error = await Effect.runPromise(
      provide(
        http,
        Effect.gen(function* () {
          const client = yield* makeIcnClient(new URL("http://127.0.0.1:8080"));
          return yield* Stream.runDrain(client.chatCompletion(request)).pipe(
            Effect.flip
          );
        })
      )
    );
    expect(error).toBeInstanceOf(IcnClientError);
    expect(error.reason).toBe("remote");
    expect(error.message).toBe("busy");
  });
});
