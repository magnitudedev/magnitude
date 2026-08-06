import { describe, expect, it } from "vitest";
import {
  AcnIdentitySchema,
  AcnInstanceIdSchema,
  AcnReady,
  AcnStarting,
  ProcessStartIdentitySchema,
  type AcnHealthState,
  type AcnInstance,
} from "@magnitudedev/acn-protocol";
import { Effect, Option, Stream } from "effect";
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
  const launches: Array<string> = [];
  const terminated: Array<string> = [];
  const manager = AcnProcessManager.of({
    observeCurrent: Effect.sync(() => {
      const value =
        options.observations[
          Math.min(observationCalls, options.observations.length - 1)
        ] ?? Option.none();
      observationCalls++;
      return value;
    }),
    launch: (request) =>
      Stream.suspend(() => {
        launches.push(request.identity);
        return options.launched === undefined
          ? Stream.dieMessage("Unexpected launch")
          : Stream.succeed({ _tag: "Ready", instance: options.launched });
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
  };
};

describe("AcnJitRuntime", () => {
  it("selects a ready current instance without launching", async () => {
    const current = instance("current");
    const fake = fakeManager({ observations: [Option.some(current)] });
    const runtime = await Effect.runPromise(
      makeAcnJitRuntime().pipe(
        Effect.provideService(AcnProcessManager, fake.manager)
      )
    );

    await Effect.runPromise(runtime.startup.prepare);
    expect(await Effect.runPromise(runtime.startup.state.get)).toEqual({
      _tag: "Ready",
    });
    expect(await Effect.runPromise(runtime.identity)).toBe(SDK_VERSION);
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
    const runtime = await Effect.runPromise(
      makeAcnJitRuntime().pipe(
        Effect.provideService(AcnProcessManager, fake.manager)
      )
    );

    await Effect.runPromise(runtime.startup.prepare);
    expect(await Effect.runPromise(runtime.identity)).toBe(newer);
    expect(fake.launches).toEqual([]);
  });

  it("launches the adopted identity after the selected instance is lost", async () => {
    const newer = AcnIdentitySchema.make("999.0.0");
    const current = instance("newer", newer);
    const replacement = instance("replacement", newer);
    const fake = fakeManager({
      observations: [Option.some(current), Option.none(), Option.none()],
      launched: replacement,
    });
    const runtime = await Effect.runPromise(
      makeAcnJitRuntime().pipe(
        Effect.provideService(AcnProcessManager, fake.manager)
      )
    );
    await Effect.runPromise(runtime.startup.prepare);
    await Effect.runPromise(runtime.startup.retry);

    expect(fake.launches).toEqual([newer]);
    expect(await Effect.runPromise(runtime.identity)).toBe(newer);
  });
});
