import * as HttpClient from "@effect/platform/HttpClient";
import type * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import * as HttpClientResponse from "@effect/platform/HttpClientResponse";
import { Chunk, Effect, Schema, Stream } from "effect";
import { describe, expect, it } from "vitest";
import {
  GeneratedClientIncompleteStreamError,
  GeneratedClientRemoteError,
  makeHttpOperation,
  makeStreamOperation,
} from "../src/client-runtime.js";

const event = Schema.Struct({ value: Schema.Number });
const remote = Schema.Struct({ code: Schema.String });
const descriptor = {
  operationId: "fixtureStream",
  transport: "sse",
  method: "POST",
  path: "/models/{model_id}/events",
  mediaType: "text/event-stream",
  payloadMediaType: "application/json",
  responseStatus: 200,
  eventSchema: event,
  termination: { type: "sentinel", value: "[DONE]" },
  reconnect: { type: "none" },
  pathParameters: Schema.Struct({ model_id: Schema.String }),
  queryParameters: Schema.Struct({ follow: Schema.BooleanFromString }),
  payload: Schema.Struct({ count: Schema.Number }),
  errors: [{ status: 409, schema: remote, mediaType: "application/json" }],
} as const;

describe("generated stream client runtime", () => {
  it("executes ordinary descriptors with the same typed response boundary", async () => {
    const operation = {
      operationId: "getFixture",
      transport: "http",
      method: "GET",
      path: "/fixtures/{id}",
      pathParameters: Schema.Struct({ id: Schema.String }),
      successes: [
        {
          status: 200,
          schema: event,
          mediaType: "application/json",
        },
      ],
      errors: [{ status: 404, schema: remote, mediaType: "application/json" }],
    } as const;
    const http = HttpClient.make((request) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          request,
          new Response('{"value":3}', {
            status: 200,
            headers: { "content-type": "application/json" },
          })
        )
      )
    );
    const call = makeHttpOperation(
      http,
      { baseUrl: "http://127.0.0.1:1" },
      operation
    );
    await expect(
      Effect.runPromise(call({ path: { id: "fixture" } }))
    ).resolves.toEqual({ value: 3 });
  });

  it("encodes inputs and decodes fragmented CRLF SSE through the declared sentinel", async () => {
    let seen: HttpClientRequest.HttpClientRequest | undefined;
    const http = HttpClient.make((request) => {
      seen = request;
      const encoder = new TextEncoder();
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(': comment\r\ndata: {"value":'));
          controller.enqueue(encoder.encode("1}\r\n\r\ndata: [DONE]\r\n\r\n"));
          controller.close();
        },
      });
      return Effect.succeed(
        HttpClientResponse.fromWeb(
          request,
          new Response(body, {
            status: 200,
            headers: { "content-type": "text/event-stream" },
          })
        )
      );
    });
    const run = makeStreamOperation(
      http,
      { baseUrl: "http://127.0.0.1:1" },
      descriptor
    );
    const values = await Effect.runPromise(
      Stream.runCollect(
        run({
          path: { model_id: "a/b" },
          urlParams: { follow: true },
          payload: { count: 2 },
        })
      )
    );
    expect(Chunk.toReadonlyArray(values)).toEqual([{ value: 1 }]);
    expect(seen?.url).toBe("http://127.0.0.1:1/models/a%2Fb/events");
    expect(seen?.urlParams).toEqual([["follow", "true"]]);
    expect(seen?.body._tag).toBe("Uint8Array");
    if (seen?.body._tag === "Uint8Array")
      expect(JSON.parse(new TextDecoder().decode(seen.body.body))).toEqual({
        count: 2,
      });
  });

  it("distinguishes declared remote errors and incomplete sentinel streams", async () => {
    const conflict = HttpClient.make((request) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          request,
          new Response('{"code":"busy"}', {
            status: 409,
            headers: { "content-type": "application/json" },
          })
        )
      )
    );
    const call = makeStreamOperation(
      conflict,
      { baseUrl: "http://127.0.0.1:1" },
      descriptor
    );
    const error = await Effect.runPromise(
      Stream.runDrain(
        call({
          path: { model_id: "model" },
          urlParams: { follow: false },
          payload: { count: 1 },
        })
      ).pipe(Effect.flip)
    );
    expect(error).toBeInstanceOf(GeneratedClientRemoteError);
    expect(
      (error as GeneratedClientRemoteError<{ code: string }>).body
    ).toEqual({ code: "busy" });

    const truncated = HttpClient.make((request) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          request,
          new Response('data: {"value":1}\n\n', {
            status: 200,
            headers: { "content-type": "text/event-stream" },
          })
        )
      )
    );
    const truncatedCall = makeStreamOperation(
      truncated,
      { baseUrl: "http://127.0.0.1:1" },
      descriptor
    );
    const incomplete = await Effect.runPromise(
      Stream.runDrain(
        truncatedCall({
          path: { model_id: "model" },
          urlParams: { follow: false },
          payload: { count: 1 },
        })
      ).pipe(Effect.flip)
    );
    expect(incomplete).toBeInstanceOf(GeneratedClientIncompleteStreamError);
  });

  it("reconnects declared long-lived SSE streams with Last-Event-ID", async () => {
    const headers: Array<string | undefined> = [];
    let requestCount = 0;
    const http = HttpClient.make((request) => {
      headers.push(request.headers["last-event-id"]);
      requestCount += 1;
      return Effect.succeed(
        HttpClientResponse.fromWeb(
          request,
          new Response(
            `id: ${requestCount}\ndata: {"value":${requestCount}}\n\n`,
            { status: 200, headers: { "content-type": "text/event-stream" } }
          )
        )
      );
    });
    const watching = {
      ...descriptor,
      operationId: "watchFixture",
      termination: { type: "long-lived" },
      reconnect: { type: "last-event-id" },
    } as const;
    const call = makeStreamOperation(
      http,
      { baseUrl: "http://127.0.0.1:1" },
      watching
    );
    const values = await Effect.runPromise(
      call({
        path: { model_id: "model" },
        urlParams: { follow: true },
        payload: { count: 1 },
      }).pipe(Stream.take(2), Stream.runCollect)
    );
    expect(Chunk.toReadonlyArray(values)).toEqual([{ value: 1 }, { value: 2 }]);
    expect(headers).toEqual([undefined, "1"]);
  });
});
