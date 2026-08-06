import { describe, expect, it } from "vitest";
import {
  AcnIdentitySchema,
  AcnInstanceIdSchema,
  AcnReady,
  AcnStarting,
  MagnitudeRpcs,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  ProcessStartIdentitySchema,
  type AcnHealthState,
  type AcnInstance,
} from "@magnitudedev/acn-protocol";
import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import * as HttpClientResponse from "@effect/platform/HttpClientResponse";
import { Rpc, RpcClient } from "@effect/rpc";
import { Effect, Exit, Layer, Option, Schema, Stream } from "effect";
import {
  AcnProcessManager,
  type AcnProcessManager as Manager,
} from "./acn-process-manager";
import { makeAcnJitRuntime } from "./acn-recovering-client";
import { SDK_VERSION } from "../version";

const instance = (
  id: string,
  identity = SDK_VERSION,
  lifecycle: AcnHealthState = new AcnReady({})
): AcnInstance => ({
  id: AcnInstanceIdSchema.make(id),
  identity,
  url: `http://${id}`,
  pid: 123,
  processStartIdentity: ProcessStartIdentitySchema.make(`process-${id}`),
  lifecycle,
});

const fakeManager = (options: {
  readonly observations: ReadonlyArray<Option.Option<AcnInstance>>;
  readonly launched?: AcnInstance;
}) => {
  let observationCalls = 0;
  let observations = options.observations;
  let published: Option.Option<AcnInstance> = Option.none();
  const launches: Array<string> = [];
  const terminated: Array<string> = [];
  const manager = AcnProcessManager.of({
    observeCurrent: Effect.sync(() => {
      if (Option.isSome(published)) return published;
      const value =
        observations[
          Math.min(observationCalls, observations.length - 1)
        ] ?? Option.none();
      observationCalls++;
      return value;
    }),
    launch: (request) =>
      Stream.suspend(() => {
        launches.push(request.identity);
        if (options.launched === undefined) {
          return Stream.dieMessage("Unexpected launch");
        }
        published = Option.some(options.launched);
        return Stream.succeed({ _tag: "Ready", instance: options.launched });
      }),
    terminate: (value) =>
      Effect.sync(() => {
        terminated.push(value.id);
      }),
  } satisfies Manager);
  return {
    manager,
    launches,
    terminated,
    observationCalls: () => observationCalls,
    setObservations: (next: ReadonlyArray<Option.Option<AcnInstance>>) => {
      observations = next;
      observationCalls = 0;
    },
  };
};

const waitForTag = (tags: ReadonlyArray<string>, tag: string) =>
  Effect.gen(function* () {
    while (!tags.includes(tag)) yield* Effect.sleep("1 millis");
  }).pipe(Effect.timeout("1 second"));

const requestBodyText = (request: HttpClientRequest.HttpClientRequest): string => {
  const body = request.body;
  if (body._tag === "Uint8Array") return new TextDecoder().decode(body.body);
  if (body._tag === "Raw" && typeof body.body === "string") return body.body;
  throw new Error(`Unexpected request body ${body._tag}`);
};

const makeRpcHttpClient = (
  current: AcnInstance,
  tags: string[],
  failRpcTags: ReadonlySet<string> = new Set()
) =>
  HttpClient.make((request) =>
    Effect.sync(() => {
      const message = JSON.parse(requestBodyText(request).split("\n")[0]) as {
        readonly id: string;
        readonly tag: string;
      };
      tags.push(message.tag);
      const rpc = MagnitudeRpcs.requests.get(message.tag);
      if (rpc === undefined) throw new Error(`Unknown RPC ${message.tag}`);
      const success =
        message.tag === "Health"
          ? {
              service: "magnitude-acn",
              version: current.identity,
              id: current.id,
              pid: current.pid,
              state: current.lifecycle,
            }
          : message.tag === "RenewClientLease"
            ? { connectedClientCount: 1 }
            : message.tag === "ReleaseClientLease"
              ? { connectedClientCount: 0 }
              : message.tag === "GetModelSlots"
                ? {
                    revision: 0,
                    state: {
                      slots: {
                        primary: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
                        secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
                      },
                      recentModelIds: { primary: [], secondary: [] },
                      favoriteModels: [],
                    },
                  }
              : undefined;
      if (success === undefined) throw new Error(`No fake response for ${message.tag}`);
      const rpcExit = failRpcTags.has(message.tag)
        ? Exit.die(`Simulated ${message.tag} failure`)
        : Exit.succeed(success);
      const exit = Schema.encodeUnknownSync(Rpc.exitSchema(rpc))(rpcExit);
      const response = `${JSON.stringify({
        _tag: "Exit",
        requestId: message.id,
        exit,
      })}\n`;
      return HttpClientResponse.fromWeb(request, new Response(response, { status: 200 }));
    })
  );

describe("AcnJitRuntime", () => {
  it("keeps lease and application RPC receivers independent", async () => {
    const current = instance("current");
    const fake = fakeManager({ observations: [Option.some(current)] });
    const tags: string[] = [];
    const http = makeRpcHttpClient(current, tags);

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const runtime = yield* makeAcnJitRuntime().pipe(
            Effect.provideService(AcnProcessManager, fake.manager),
            Effect.provideService(HttpClient.HttpClient, http)
          );
          const applicationClient = yield* RpcClient.make(MagnitudeRpcs).pipe(
            Effect.provide(
              runtime.protocolLayer.pipe(
                Layer.provide(Layer.succeed(HttpClient.HttpClient, http))
              )
            )
          );
          const secondApplicationClient = yield* RpcClient.make(MagnitudeRpcs).pipe(
            Effect.provide(
              runtime.protocolLayer.pipe(
                Layer.provide(Layer.succeed(HttpClient.HttpClient, http))
              )
            )
          );
          const health = yield* Effect.all(
            [applicationClient.Health({}), secondApplicationClient.Health({})],
            { concurrency: "unbounded" }
          ).pipe(Effect.timeout("1 second"));
          expect(health.map((response) => response.id)).toEqual([
            current.id,
            current.id,
          ]);
          yield* Effect.yieldNow();
          expect(tags).toContain("RenewClientLease");
          expect(tags.filter((tag) => tag === "Health")).toHaveLength(2);
          expect(Option.isSome(yield* runtime.close)).toBe(true);
          expect(Option.isSome(yield* runtime.close)).toBe(true);
          expect(tags.filter((tag) => tag === "ReleaseClientLease")).toHaveLength(1);
          const healthAfterClose = yield* Effect.exit(applicationClient.Health({}));
          expect(Exit.isFailure(healthAfterClose)).toBe(true);
          expect(tags.filter((tag) => tag === "Health")).toHaveLength(2);
        })
      )
    );
  });

  it("releases the lease when close observation fails", async () => {
    const current = instance("current");
    const fake = fakeManager({ observations: [Option.some(current)] });
    const tags: string[] = [];
    const http = makeRpcHttpClient(
      current,
      tags,
      new Set(["GetModelSlots"])
    );

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const runtime = yield* makeAcnJitRuntime().pipe(
            Effect.provideService(AcnProcessManager, fake.manager),
            Effect.provideService(HttpClient.HttpClient, http)
          );
          yield* waitForTag(tags, "RenewClientLease");
          expect(Option.isNone(yield* runtime.close)).toBe(true);
          expect(tags).toContain("GetModelSlots");
          expect(tags.filter((tag) => tag === "ReleaseClientLease")).toHaveLength(1);
        })
      )
    );
  });

  it("selects a ready current instance without launching", async () => {
    const current = instance("current");
    const fake = fakeManager({ observations: [Option.some(current)] });
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const runtime = yield* makeAcnJitRuntime().pipe(
            Effect.provideService(AcnProcessManager, fake.manager),
            Effect.provideService(
              HttpClient.HttpClient,
              makeRpcHttpClient(current, [])
            )
          );
          yield* runtime.startup.prepare;
          expect(yield* runtime.startup.state.get).toEqual({ _tag: "Ready" });
          expect(yield* runtime.identity).toBe(SDK_VERSION);
        })
      )
    );
    expect(fake.launches).toEqual([]);
  });

  it("adopts a newer identity while that exact instance is starting", async () => {
    const newer = AcnIdentitySchema.make("999.0.0");
    const starting = instance(
      "newer",
      newer,
      new AcnStarting({
        activity: "Resolving",
        progress: Option.none(),
      })
    );
    const ready = { ...starting, lifecycle: new AcnReady({}) };
    const fake = fakeManager({
      observations: [Option.some(starting), Option.some(ready)],
    });
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const runtime = yield* makeAcnJitRuntime().pipe(
            Effect.provideService(AcnProcessManager, fake.manager),
            Effect.provideService(
              HttpClient.HttpClient,
              makeRpcHttpClient(ready, [])
            )
          );
          yield* runtime.startup.prepare;
          expect(yield* runtime.identity).toBe(newer);
        })
      )
    );
    expect(fake.launches).toEqual([]);
  });

  it("launches the adopted identity after the selected instance is lost", async () => {
    const newer = AcnIdentitySchema.make("999.0.0");
    const current = instance("newer", newer);
    const replacement = instance("replacement", newer);
    const fake = fakeManager({
      observations: [Option.some(current)],
      launched: replacement,
    });
    const tags: string[] = [];
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const runtime = yield* makeAcnJitRuntime().pipe(
            Effect.provideService(AcnProcessManager, fake.manager),
            Effect.provideService(
              HttpClient.HttpClient,
              makeRpcHttpClient(current, tags)
            )
          );
          yield* runtime.startup.prepare.pipe(Effect.timeout("1 second"));
          yield* waitForTag(tags, "RenewClientLease");
          fake.setObservations([Option.none()]);
          yield* runtime.startup.retry.pipe(Effect.timeout("4 seconds"));
          expect(yield* runtime.identity).toBe(newer);
        })
      )
    );

    expect(fake.launches).toEqual([newer]);
  }, 10_000);

  it("close never launches a missing ACN", async () => {
    const current = instance("current");
    const replacement = instance("replacement");
    const fake = fakeManager({
      observations: [Option.some(current)],
      launched: replacement,
    });
    const tags: string[] = [];

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const runtime = yield* makeAcnJitRuntime().pipe(
            Effect.provideService(AcnProcessManager, fake.manager),
            Effect.provideService(
              HttpClient.HttpClient,
              makeRpcHttpClient(current, tags)
            )
          );
          yield* runtime.startup.prepare.pipe(Effect.timeout("1 second"));
          yield* waitForTag(tags, "RenewClientLease");
          fake.setObservations([Option.none()]);

          expect(Option.isSome(yield* runtime.close)).toBe(true);
          expect(Exit.isFailure(yield* Effect.exit(runtime.startup.retry))).toBe(true);
          expect(fake.observationCalls()).toBe(0);
          expect(fake.launches).toEqual([]);
        })
      )
    );

    expect(fake.launches).toEqual([]);
  });

  it("scope finalization stops renewal without launching or releasing", async () => {
    const current = instance("current");
    const replacement = instance("replacement");
    const fake = fakeManager({
      observations: [Option.some(current)],
      launched: replacement,
    });
    const tags: string[] = [];
    const http = makeRpcHttpClient(current, tags);

    const protocolLayer = await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const runtime = yield* makeAcnJitRuntime().pipe(
            Effect.provideService(AcnProcessManager, fake.manager),
            Effect.provideService(HttpClient.HttpClient, http)
          );
          yield* waitForTag(tags, "RenewClientLease");
          fake.setObservations([Option.none()]);
          return runtime.protocolLayer;
        })
      )
    );

    const postFinalization = await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const applicationClient = yield* RpcClient.make(MagnitudeRpcs).pipe(
            Effect.provide(
              protocolLayer.pipe(
                Layer.provide(Layer.succeed(HttpClient.HttpClient, http))
              )
            )
          );
          return yield* Effect.exit(applicationClient.Health({}));
        })
      )
    );

    expect(Exit.isFailure(postFinalization)).toBe(true);
    expect(fake.observationCalls()).toBe(0);
    expect(fake.launches).toEqual([]);
    expect(tags).not.toContain("ReleaseClientLease");
  });
});
