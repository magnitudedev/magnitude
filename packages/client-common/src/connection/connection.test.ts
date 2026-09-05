import {
  HttpClient,
  HttpClientError,
  HttpClientResponse,
} from "@effect/platform";
import {
  MagnitudeClient,
  MagnitudeServiceStarter,
  MAGNITUDE_RPC_VERSION,
} from "@magnitudedev/sdk";
import { Effect, Layer, Stream } from "effect";
import { describe, expect, it } from "vitest";
import { makeFirstPartyConnection } from "./connection";

const fixture = () => {
  let running = true;
  let id = "first";
  let starts = 0;
  let requests = 0;
  const health = () => ({
    service: "magnitude-acn",
    version: "0.0.1",
    revision: 1,
    rpcVersion: MAGNITUDE_RPC_VERSION,
    id,
    pid: 123,
    state: { _tag: "Ready" },
  });
  const http = HttpClient.make((request) =>
    Effect.suspend(() => {
      requests++;
      if (!running)
        return Effect.fail(
          new HttpClientError.RequestError({ request, reason: "Transport" })
        );
      if (request.url.endsWith("/health"))
        return Effect.succeed(
          HttpClientResponse.fromWeb(
            request,
            Response.json({
              service: "magnitude-acn",
              version: "0.0.1",
              revision: 1,
              rpcVersion: MAGNITUDE_RPC_VERSION,
              id,
              pid: 123,
              state: { _tag: "Ready" },
            })
          )
        );
      if (request.headers["x-magnitude-acn-id"] !== id)
        return Effect.succeed(
          HttpClientResponse.fromWeb(
            request,
            new Response(null, { status: 409 })
          )
        );
      if (request.body._tag !== "Uint8Array")
        throw new Error("Expected RPC body");
      const input = JSON.parse(new TextDecoder().decode(request.body.body));
      return Effect.succeed(
        HttpClientResponse.fromWeb(
          request,
          new Response(
            JSON.stringify({
              _tag: "Exit",
              requestId: input.id,
              exit: { _tag: "Success", value: health() },
            }) + "\n"
          )
        )
      );
    })
  );
  const layer = MagnitudeClient.layer().pipe(
    Layer.provide([
      Layer.succeed(HttpClient.HttpClient, http),
      Layer.succeed(MagnitudeServiceStarter, {
        start: Stream.execute(
          Effect.sync(() => {
            starts++;
            running = true;
            id = "replacement";
          })
        ),
      }),
    ])
  );
  return {
    layer,
    requests: () => requests,
    starts: () => starts,
    stop: () => {
      running = false;
    },
  };
};

describe("first-party SDK presentation", () => {
  it("is passive and owns one SDK scope without owning the daemon", async () => {
    const f = fixture();
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const connection = yield* makeFirstPartyConnection(f.layer);
          expect((yield* connection.startup.state.get)._tag).toBe("Checking");
          expect(f.requests()).toBe(0);
          yield* connection.startup.awaitReady;
          yield* connection.startup.state.changes.pipe(
            Stream.filter((state) => state._tag === "Ready"),
            Stream.runHead
          );
          expect(f.starts()).toBe(0);
          yield* connection.close;
          yield* connection.close;
          expect((yield* connection.client.connection.state)._tag).toBe(
            "Closed"
          );
          expect(
            (yield* Effect.either(connection.client.connection.connect))._tag
          ).toBe("Left");
        })
      )
    );
  });
  it("keeps initial readiness and the same SDK instance during service recovery", async () => {
    const f = fixture();
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const connection = yield* makeFirstPartyConnection(f.layer);
          const client = connection.client;
          yield* connection.startup.awaitReady;
          yield* connection.startup.state.changes.pipe(
            Stream.filter((state) => state._tag === "Ready"),
            Stream.runHead
          );
          f.stop();
          // Health is replay-safe; StopActiveLocalModel is deliberately not.
          yield* client.connection.health({});
          yield* connection.startup.recovery.changes.pipe(
            Stream.filter((state) => state._tag === "Recovered"),
            Stream.runHead
          );
          expect(connection.client).toBe(client);
          expect((yield* connection.startup.state.get)._tag).toBe("Ready");
          expect(f.starts()).toBe(1);
        })
      )
    );
  });
});
