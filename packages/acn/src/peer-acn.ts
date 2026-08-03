import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import type * as FileSystem from "@effect/platform/FileSystem";
import { Clock, Data, Effect, Option } from "effect";
import type { AcnOwnerId } from "@magnitudedev/acn-protocol";
import {
  listAcnInstances,
  readRegistrationOwnership,
  registrationIsOwnedBy,
  registrationPath,
  removeExactInstance,
  type RegisteredAcnInstance,
} from "./daemon-registration";
import { readProcessStartIdentity } from "./process-identity";

export class AcnPeerRemovalFailed extends Data.TaggedError("AcnPeerRemovalFailed")<{
  readonly id: AcnOwnerId;
  readonly reason: string;
}> {}

const isExactProcess = (instance: RegisteredAcnInstance) =>
  readProcessStartIdentity(instance.record.pid).pipe(
    Effect.map((identity) =>
      Option.exists(
        identity,
        (identity) => identity === instance.record.processStartIdentity,
      ),
    ),
  );

const isMissingProcessError = (error: unknown): boolean =>
  error instanceof Error && "code" in error && error.code === "ESRCH";

const isProcessAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !isMissingProcessError(error);
  }
};

const sendSignal = (pid: number, signal: NodeJS.Signals): boolean => {
  try {
    process.kill(pid, signal);
    return true;
  } catch (error) {
    if (isMissingProcessError(error)) return false;
    throw error;
  }
};

const waitForExit = (pid: number, durationMs: number) =>
  Effect.gen(function* () {
    const deadline = (yield* Clock.currentTimeMillis) + durationMs;
    while ((yield* Clock.currentTimeMillis) < deadline) {
      if (!isProcessAlive(pid)) return true;
      yield* Effect.sleep("50 millis");
    }
    return !isProcessAlive(pid);
  });

const signalExact = (
  instance: RegisteredAcnInstance,
  signal: NodeJS.Signals,
  assertCanonical: Effect.Effect<void, AcnPeerRemovalFailed, FileSystem.FileSystem>,
) =>
  Effect.gen(function* () {
    yield* assertCanonical;
    if (!(yield* isExactProcess(instance))) return false;
    return yield* Effect.try({
      try: () => sendSignal(instance.record.pid, signal),
      catch: (error) => new AcnPeerRemovalFailed({
        id: instance.record.id,
        reason: `Unable to send ${signal}: ${String(error)}`,
      }),
    });
  });

export const terminatePublishedAcn = (
  instance: RegisteredAcnInstance,
  authorize: Effect.Effect<void, AcnPeerRemovalFailed, FileSystem.FileSystem> = Effect.void,
) =>
  Effect.gen(function* () {
    yield* authorize;
    if (!(yield* isExactProcess(instance))) {
      yield* removeExactInstance(instance);
      return;
    }

    const client = yield* HttpClient.HttpClient;
    if (Option.isSome(instance.record.url)) {
      // Process inspection can block. Reauthorize after it and immediately
      // before asking the peer to stop.
      yield* authorize;
      yield* client.execute(
        HttpClientRequest.post(`${instance.record.url.value}/shutdown`).pipe(
          HttpClientRequest.setHeader("x-magnitude-acn-id", instance.record.id),
        ),
      ).pipe(Effect.timeout("750 millis"), Effect.ignore);
      if (yield* waitForExit(instance.record.pid, 2_000)) {
        yield* removeExactInstance(instance);
        return;
      }
    }

    if (!(yield* signalExact(instance, "SIGINT", authorize))) {
      yield* removeExactInstance(instance);
      return;
    }
    if (yield* waitForExit(instance.record.pid, 2_000)) {
      yield* removeExactInstance(instance);
      return;
    }

    if (!(yield* signalExact(instance, "SIGKILL", authorize))) {
      yield* removeExactInstance(instance);
      return;
    }
    if (!(yield* waitForExit(instance.record.pid, 2_000))) {
      return yield* new AcnPeerRemovalFailed({
        id: instance.record.id,
        reason: "Exact peer remained alive after forceful termination",
      });
    }
    yield* removeExactInstance(instance);
  }).pipe(
    Effect.mapError((error) =>
      error instanceof AcnPeerRemovalFailed
        ? error
        : new AcnPeerRemovalFailed({
            id: instance.record.id,
            reason: `Unable to inspect exact ACN process: ${String(error)}`,
          }),
    ),
  );

export const removePeerAcns = (dataDir: string, selfId: AcnOwnerId) => {
  const assertCanonical: Effect.Effect<
    void,
    AcnPeerRemovalFailed,
    FileSystem.FileSystem
  > = readRegistrationOwnership(registrationPath(dataDir)).pipe(
    Effect.mapError((error) =>
      new AcnPeerRemovalFailed({
        id: selfId,
        reason: `Unable to verify canonical ownership: ${String(error)}`,
      }),
    ),
    Effect.filterOrFail(
      (registration) => registrationIsOwnedBy(registration, selfId),
      () =>
        new AcnPeerRemovalFailed({
          id: selfId,
          reason: "ACN lost canonical ownership during peer removal",
        }),
    ),
    Effect.asVoid,
  );
  return listAcnInstances(dataDir).pipe(
    Effect.flatMap((instances) =>
      Effect.gen(function* () {
        const peers = instances.filter(({ record }) => record.id !== selfId);
        const results = yield* Effect.forEach(
          peers,
          (instance) => terminatePublishedAcn(instance, assertCanonical).pipe(
            Effect.either,
          ),
          { concurrency: "unbounded" },
        );
        const failed = results.find((result) => result._tag === "Left");
        if (failed?._tag === "Left") return yield* failed.left;
      }),
    ),
    Effect.mapError((error) =>
      error instanceof AcnPeerRemovalFailed
        ? error
        : new AcnPeerRemovalFailed({
            id: selfId,
            reason: `Unable to enumerate or clean ACN peers: ${String(error)}`,
          }),
    ),
  );
};
