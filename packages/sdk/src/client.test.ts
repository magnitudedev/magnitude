import { describe, expect, it } from "vitest";
import {
  HttpClient,
  HttpClientError,
  HttpClientResponse,
} from "@effect/platform";
import {
  Context,
  Deferred,
  Effect,
  Exit,
  Fiber,
  Layer,
  Scope,
  Stream,
} from "effect";
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/acn-protocol";
import { MagnitudeClient } from "./client";
import { MagnitudeServiceStarter } from "./service-starter";
import { ServiceStartFailed } from "./connection-errors";

const health = (id = "first", rpcVersion = MAGNITUDE_RPC_VERSION) => ({
  service: "magnitude-acn",
  version: "0.0.1",
  revision: 1,
  rpcVersion,
  id,
  pid: 123,
  state: { _tag: "Ready" },
});

/** A daemon from before the RPC version existed: the same health, minus the field. */
const legacyHealth = (id: string) => {
  const { rpcVersion: _, ...rest } = health(id);
  return rest;
};

const fixture = (options: { readonly starterUpgrades?: boolean } = {}) => {
  let running = true;
  let version = MAGNITUDE_RPC_VERSION;
  let legacy = false;
  let id = "first";
  const requests: string[] = [];
  let starts = 0;
  const http = HttpClient.make((request) =>
    Effect.suspend(() => {
      requests.push(request.url);
      if (!running)
        return Effect.fail(
          new HttpClientError.RequestError({ request, reason: "Transport" })
        );
      if (request.url.endsWith("/health"))
        return Effect.succeed(
          HttpClientResponse.fromWeb(
            request,
            Response.json(legacy ? legacyHealth(id) : health(id, version))
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
        throw new Error("Expected encoded RPC body");
      const message = JSON.parse(
        new TextDecoder().decode(request.body.body).trim()
      );
      return Effect.succeed(
        HttpClientResponse.fromWeb(
          request,
          new Response(
            JSON.stringify({
              _tag: "Exit",
              requestId: message.id,
              exit: { _tag: "Success", value: {} },
            }) + "\n"
          )
        )
      );
    })
  );
  const start = Stream.execute(
    Effect.sync(() => {
      starts++;
      running = true;
      // A privileged starter replaces an older or mismatched daemon with the current one.
      if (options.starterUpgrades) {
        legacy = false;
        version = MAGNITUDE_RPC_VERSION;
      }
    })
  );
  const live = MagnitudeClient.layer().pipe(
    Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
    Layer.provide(Layer.succeed(MagnitudeServiceStarter, { start }))
  );
  return {
    live,
    http,
    requests,
    starts: () => starts,
    stop: () => {
      running = false;
    },
    resume: () => {
      running = true;
    },
    replace: () => {
      id = "second";
    },
    mismatch: () => {
      version++;
    },
    legacy: () => {
      legacy = true;
    },
  };
};

describe("MagnitudeClient", () => {
  it("scope closure cancels startup, terminalizes waiters, and prevents reuse", async () => {
    const f = fixture();
    f.stop();
    await Effect.runPromise(
      Effect.gen(function* () {
        const entered = yield* Deferred.make<void>();
        const cancelled = yield* Deferred.make<void>();
        const scope = yield* Scope.make();
        const live = MagnitudeClient.layer().pipe(
          Layer.provide([
            Layer.succeed(HttpClient.HttpClient, f.http),
            Layer.succeed(MagnitudeServiceStarter, {
              start: Stream.execute(
                Deferred.succeed(entered, undefined).pipe(
                  Effect.zipRight(Effect.never),
                  Effect.ensuring(Deferred.succeed(cancelled, undefined))
                )
              ),
            }),
          ])
        );
        const client = Context.get(
          yield* Layer.buildWithScope(live, scope),
          MagnitudeClient
        );
        const pending = yield* client.connection.connect.pipe(Effect.fork);
        yield* Deferred.await(entered);
        yield* Scope.close(scope, Exit.void);
        yield* Deferred.await(cancelled);
        expect((yield* Fiber.await(pending))._tag).toBe("Failure");
        expect((yield* client.connection.state)._tag).toBe("Closed");
        const result = yield* Effect.either(client.connection.connect);
        expect(result._tag === "Left" && result.left._tag).toBe(
          "ConnectionClosed"
        );
      })
    );
  });

  it("shares a starter failure and can retry without rebuilding the client", async () => {
    const f = fixture();
    f.stop();
    let attempts = 0;
    const start = Stream.execute(
      Effect.suspend(() => {
        attempts++;
        return attempts === 1
          ? Effect.fail(new ServiceStartFailed({ message: "deliberate" }))
          : Effect.sync(f.resume);
      })
    );
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        const failures = yield* Effect.all(
          [client.connection.connect, client.connection.connect].map(
            Effect.either
          ),
          { concurrency: "unbounded" }
        );
        expect(attempts).toBe(1);
        expect(
          failures.every(
            (result) =>
              result._tag === "Left" &&
              result.left._tag === "ServiceStartFailed"
          )
        ).toBe(true);
        yield* client.connection.connect;
        expect(attempts).toBe(2);
      }).pipe(
        Effect.provide(
          MagnitudeClient.layer().pipe(
            Layer.provide([
              Layer.succeed(HttpClient.HttpClient, f.http),
              Layer.succeed(MagnitudeServiceStarter, { start }),
            ])
          )
        )
      )
    );
  });

  it("starter completion cannot admit RPC without ready health", async () => {
    const f = fixture();
    f.stop();
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        const result = yield* Effect.either(client.models.stop({}));
        expect(result._tag === "Left" && result.left._tag).toBe(
          "ServiceUnavailable"
        );
        expect(f.requests.every((url) => url.endsWith("/health"))).toBe(true);
        expect((yield* client.connection.state)._tag).toBe("Failed");
      }).pipe(
        Effect.provide(
          MagnitudeClient.layer({ connectTimeout: "30 millis" }).pipe(
            Layer.provide([
              Layer.succeed(HttpClient.HttpClient, f.http),
              Layer.succeed(MagnitudeServiceStarter, { start: Stream.empty }),
            ])
          )
        )
      )
    );
  });

  it("construction and observation do not contact or start the service", async () => {
    const f = fixture();
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        expect((yield* client.connection.state)._tag).toBe("Idle");
        expect(
          (yield* client.connection.changes.pipe(Stream.runHead))._tag
        ).toBe("Some");
        expect(f.requests).toEqual([]);
        expect(f.starts()).toBe(0);
      }).pipe(Effect.provide(f.live))
    );
  });

  it("starts once for concurrent callers, admits RPC, and does not stop on close", async () => {
    const f = fixture();
    f.stop();
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        yield* Effect.all(
          Array.from({ length: 8 }, () => client.connection.connect),
          { concurrency: "unbounded" }
        );
        expect(f.starts()).toBe(1);
        yield* client.models.stop({});
        expect((yield* client.connection.state)._tag).toBe("Ready");
      }).pipe(Effect.provide(f.live))
    );
    expect(f.starts()).toBe(1);
  });

  it("gives the starter one chance at a mismatched daemon, then reports the mismatch", async () => {
    const f = fixture();
    f.mismatch();
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        const result = yield* Effect.either(client.models.stop({}));
        expect(result._tag).toBe("Left");
        if (result._tag === "Left")
          expect(result.left._tag).toBe("ProtocolMismatch");
        expect(f.starts()).toBe(1);
        expect(f.requests.every((url) => url.endsWith("/health"))).toBe(true);
      }).pipe(Effect.provide(f.live))
    );
  });

  it("replaces a daemon that predates the RPC version through the starter", async () => {
    const f = fixture({ starterUpgrades: true });
    f.legacy();
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        yield* client.models.stop({});
        expect(f.starts()).toBe(1);
        const state = yield* client.connection.state;
        expect(state._tag).toBe("Ready");
      }).pipe(Effect.provide(f.live))
    );
  });

  it("reports a pre-RPC daemon as version 0 when no starter can replace it", async () => {
    const f = fixture();
    f.legacy();
    const live = MagnitudeClient.layer({ autoStart: false }).pipe(
      Layer.provide(Layer.succeed(HttpClient.HttpClient, f.http))
    );
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        const result = yield* Effect.either(client.connection.connect);
        expect(result._tag).toBe("Left");
        if (result._tag === "Left" && result.left._tag === "ProtocolMismatch") {
          expect(result.left.expected).toBe(MAGNITUDE_RPC_VERSION);
          expect(result.left.actual).toBe(0);
        } else expect.unreachable("expected a protocol mismatch");
        expect(f.starts()).toBe(0);
      }).pipe(Effect.provide(live))
    );
  });

  it("rechecks a replacement at the same address before redispatch", async () => {
    const f = fixture();
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        yield* client.connection.connect;
        f.replace();
        f.mismatch();
        const result = yield* Effect.either(client.models.stop({}));
        expect(result._tag).toBe("Left");
        if (result._tag === "Left")
          expect(result.left._tag).toBe("ProtocolMismatch");
      }).pipe(Effect.provide(f.live))
    );
  });

  it("normalizes connection errors on streaming operations through the same boundary", async () => {
    const f = fixture();
    f.mismatch();
    await Effect.runPromise(Effect.gen(function* () {
      const client = yield* MagnitudeClient;
      const result = yield* Effect.either(client.changes.streamChanges({}).pipe(Stream.runHead));
      expect(result._tag).toBe("Left");
      if (result._tag === "Left") expect(result.left._tag).toBe("ProtocolMismatch");
      expect(f.starts()).toBe(1);
      expect(f.requests.every(url => url.endsWith("/health"))).toBe(true);
    }).pipe(Effect.provide(f.live)));
  });

  it("connect-only failure can be retried on the same client", async () => {
    const f = fixture();
    f.stop();
    const live = MagnitudeClient.layer({ autoStart: false }).pipe(
      Layer.provide(Layer.succeed(HttpClient.HttpClient, f.http))
    );
    await Effect.runPromise(
      Effect.gen(function* () {
        const client = yield* MagnitudeClient;
        expect((yield* Effect.either(client.connection.connect))._tag).toBe(
          "Left"
        );
        expect((yield* client.connection.state)._tag).toBe("Failed");
        expect(f.starts()).toBe(0);
        f.resume();
        yield* client.connection.connect;
        expect((yield* client.connection.state)._tag).toBe("Ready");
      }).pipe(Effect.provide(live))
    );
  });

  it("cancelling one waiter does not cancel the shared startup", async () => {
    const f = fixture();
    f.stop();
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const entered = yield* Deferred.make<void>();
          const release = yield* Deferred.make<void>();
          const start = Stream.execute(
            Deferred.succeed(entered, undefined).pipe(
              Effect.zipRight(Deferred.await(release)),
              Effect.tap(() => Effect.sync(f.resume))
            )
          );
          const live = MagnitudeClient.layer().pipe(
            Layer.provide(Layer.succeed(HttpClient.HttpClient, f.http)),
            Layer.provide(Layer.succeed(MagnitudeServiceStarter, { start }))
          );
          yield* Effect.gen(function* () {
            const client = yield* MagnitudeClient;
            const first = yield* client.connection.connect.pipe(Effect.fork);
            yield* Deferred.await(entered);
            const second = yield* client.connection.connect.pipe(Effect.fork);
            yield* Fiber.interrupt(first);
            yield* Deferred.succeed(release, undefined);
            yield* Fiber.join(second);
            expect((yield* client.connection.state)._tag).toBe("Ready");
          }).pipe(Effect.provide(live));
        })
      )
    );
  });
});
