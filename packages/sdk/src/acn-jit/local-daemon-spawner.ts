/**
 * Local daemon spawner — shared logic for Bun and Node environments.
 *
 * `makeLocalDaemonSpawner` is an `Effect` that captures `FileSystem`,
 * `HttpClient`, and `CommandExecutor` at construction time, returning a sealed
 * `DaemonSpawner` whose methods require only `never`. This keeps the
 * `DaemonSpawner` interface clean while the layer has the real requirements.
 *
 * The actual process spawning is delegated to the `spawnProcess` function,
 * which differs between Bun (`Bun.spawn`) and Node (`child_process.spawn`).
 */
import { Effect, Option, Schedule, Schema } from "effect";
import { FileSystem } from "@effect/platform/FileSystem";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import * as NodePath from "node:path";
import { readStructuredFile } from "@magnitudedev/storage";
import {
  AcnVersionRegistrySchema,
  type AcnRegistration,
  versionPathSegment,
} from "@magnitudedev/protocol";
import {
  DaemonCrashed,
  DaemonError,
  DaemonSpawnFailed,
  NoDaemon,
  RegistrationFileInvalid,
} from "./errors";
import { resolveBinaryCommand, defaultDataDir } from "../binary";
import { SDK_VERSION } from "../version";
import type { DaemonSpawner } from "./daemon-spawner";

/**
 * Function that spawns a detached process. Provided by the consumer —
 * `Bun.spawn` for Bun, `child_process.spawn` for Node.
 */
export interface SpawnProcess {
  (command: string[]): {
    readonly pid: number | undefined;
    readonly exited: Promise<number | null>;
  };
}

// ─── Internal types ──────────────────────────────────────────────────────────

export interface HealthResponse {
  readonly service: string;
  readonly version: string;
  readonly id: string;
  readonly pid: number;
}

const HealthResponseSchema = Schema.Struct({
  service: Schema.String,
  version: Schema.String,
  id: Schema.String,
  pid: Schema.Number,
});

type DebugField =
  | string
  | number
  | boolean
  | null
  | undefined
  | ReadonlyArray<string | number | boolean | null>;

const debugLog = (
  enabled: boolean,
  message: string,
  fields?: Readonly<Record<string, DebugField>>
): Effect.Effect<void> =>
  enabled
    ? Effect.logDebug(message).pipe(Effect.annotateLogs(fields ?? {}))
    : Effect.void;

// ─── Path helpers ────────────────────────────────────────────────────────────

const versionDirectory = (dataDir: string, version: string): string =>
  NodePath.join(dataDir, "acn", versionPathSegment(version));

const registrationPath = (dataDir: string, version: string): string =>
  NodePath.join(versionDirectory(dataDir, version), "registry.json");

const spawnElectionPath = (dataDir: string, version: string): string =>
  NodePath.join(versionDirectory(dataDir, version), "spawn-election");

const spawnElectionOwnerPath = (path: string): string =>
  NodePath.join(path, "owner");

interface SpawnElectionClaim {
  readonly path: string;
  readonly token: string;
}

const platformErrorReason = (cause: unknown): string | undefined =>
  typeof cause === "object" && cause !== null && "reason" in cause
    ? String(cause.reason)
    : undefined;

const electionFailure = (operation: string, cause: unknown) =>
  new DaemonSpawnFailed({
    reason: `${operation}: ${String(cause)}`,
  });

const tryAcquireSpawnElection = (
  path: string,
  staleAfterMs: number,
  fs: FileSystem,
): Effect.Effect<Option.Option<SpawnElectionClaim>, DaemonSpawnFailed> => Effect.gen(function* () {
  yield* fs.makeDirectory(NodePath.dirname(path), { recursive: true }).pipe(
    Effect.mapError((cause) => electionFailure("Failed to create ACN registry directory", cause)),
  );
  const acquired = yield* fs.makeDirectory(path).pipe(
    Effect.as(true),
    Effect.catchAll((cause) => platformErrorReason(cause) === "AlreadyExists"
      ? Effect.succeed(false)
      : Effect.fail(electionFailure("Failed to acquire ACN spawn election", cause))),
  );
  if (acquired) {
    const claim = { path, token: crypto.randomUUID() } satisfies SpawnElectionClaim;
    yield* fs.writeFileString(spawnElectionOwnerPath(path), claim.token).pipe(
      Effect.mapError((cause) => electionFailure("Failed to record ACN spawn election owner", cause)),
      Effect.onError(() => fs.remove(path, { recursive: true }).pipe(Effect.ignore)),
    );
    return Option.some(claim);
  }

  // A crash can leave the claim directory behind. Age-based recovery is
  // deliberately conservative: a healthy contender normally releases it as
  // soon as registration becomes observable.
  const info = yield* fs.stat(path).pipe(
    Effect.map(Option.some),
    Effect.catchAll((cause) => platformErrorReason(cause) === "NotFound"
      ? Effect.succeed(Option.none())
      : Effect.fail(electionFailure("Failed to inspect ACN spawn election", cause))),
  );
  if (Option.isSome(info)) {
    const modifiedAt = Option.getOrUndefined(info.value.mtime);
    if (modifiedAt && Date.now() - modifiedAt.getTime() > staleAfterMs) {
      yield* fs.remove(path, { recursive: true }).pipe(
        Effect.catchAll((cause) => platformErrorReason(cause) === "NotFound"
          ? Effect.void
          : Effect.fail(electionFailure("Failed to recover stale ACN spawn election", cause))),
      );
    }
  }
  return Option.none();
});

const releaseSpawnElection = (
  claim: SpawnElectionClaim,
  fs: FileSystem,
): Effect.Effect<void> => fs.readFileString(spawnElectionOwnerPath(claim.path)).pipe(
  Effect.flatMap((owner) => owner === claim.token
    ? fs.remove(claim.path, { recursive: true })
    : Effect.void),
  Effect.catchAll(() => Effect.void),
);

const withSpawnElection = <A, E>(
  path: string,
  timeoutMs: number,
  fs: FileSystem,
  effect: Effect.Effect<A, E, never>
): Effect.Effect<A, E | DaemonSpawnFailed, never> => {
  const staleAfterMs = Math.max(60_000, timeoutMs * 2);
  const retryCount = Math.max(1, Math.ceil(timeoutMs / 50));
  const acquire = tryAcquireSpawnElection(path, staleAfterMs, fs).pipe(
    Effect.filterOrFail(
      Option.isSome,
      () => new NoDaemon(),
    ),
    Effect.map((claim) => claim.value),
    Effect.retry({
      schedule: Schedule.spaced("50 millis").pipe(
        Schedule.intersect(Schedule.recurs(retryCount)),
      ),
      while: (failure) => failure._tag === "NoDaemon",
    }),
    Effect.mapError((failure) => failure instanceof NoDaemon
      ? new DaemonSpawnFailed({ reason: "Timed out waiting for ACN spawn election" })
      : failure),
  );
  return Effect.acquireUseRelease(
    acquire,
    () => effect,
    (claim) => releaseSpawnElection(claim, fs),
  );
};

// ─── Registration reading ────────────────────────────────────────────────────

const readRegistration = (
  path: string,
  fs: FileSystem
): Effect.Effect<Option.Option<AcnRegistration>, RegistrationFileInvalid> =>
  Effect.gen(function* () {
    const result = yield* readStructuredFile(
      path,
      AcnVersionRegistrySchema
    ).pipe(
      Effect.provideService(FileSystem, fs),
      Effect.mapError(
        (cause) => new RegistrationFileInvalid({ path, reason: String(cause) })
      )
    );
    if (result._tag === "Missing") return Option.none();
    if (result._tag === "Invalid") {
      yield* Effect.logWarning(
        "Ignoring invalid disposable ACN registration"
      ).pipe(Effect.annotateLogs({ path, reason: result.error.reason }));
      return Option.none();
    }
    return Option.fromNullable(result.value.registration);
  });

// ─── Health probing ──────────────────────────────────────────────────────────

const probeHealth = (
  url: string,
  timeoutMs: number,
  client: HttpClient.HttpClient
): Effect.Effect<HealthResponse, NoDaemon, never> =>
  Effect.gen(function* () {
    const response = yield* client
      .execute(HttpClientRequest.get(`${url}/health`))
      .pipe(
        Effect.timeout(`${timeoutMs} millis`),
        Effect.mapError(() => new NoDaemon())
      );

    if (response.status < 200 || response.status >= 300) {
      return yield* new NoDaemon();
    }

    const json = yield* response.json.pipe(
      Effect.mapError(() => new NoDaemon())
    );
    const health = yield* Schema.decodeUnknown(HealthResponseSchema)(json).pipe(
      Effect.mapError(() => new NoDaemon())
    );

    if (health.service !== "magnitude-acn") {
      return yield* new NoDaemon();
    }

    return health;
  });

// ─── Spawn daemon (wait-for-registration pipeline) ───────────────────────────

const spawnDaemon = (
  command: string[],
  options: {
    readonly dataDir: string;
    readonly version: string;
    readonly timeoutMs: number;
    readonly debug: boolean;
  },
  deps: {
    readonly fs: FileSystem;
    readonly client: HttpClient.HttpClient;
    readonly spawnProcess: SpawnProcess;
  }
): Effect.Effect<
  string,
  DaemonSpawnFailed | DaemonCrashed | RegistrationFileInvalid,
  never
> =>
  Effect.gen(function* () {
    const { fs, client, spawnProcess } = deps;
    yield* debugLog(options.debug, "spawning ACN", {
      command: command.join(" "),
      detached: true,
    });

    const proc = spawnProcess(command);

    yield* debugLog(options.debug, "ACN process spawned", { pid: proc.pid });

    const regPath = registrationPath(options.dataDir, options.version);

    const checkHealthyRegistration: Effect.Effect<
      AcnRegistration,
      NoDaemon | RegistrationFileInvalid,
      never
    > = Effect.gen(function* () {
      const registrationOption = yield* readRegistration(regPath, fs);
      if (Option.isNone(registrationOption)) return yield* new NoDaemon();

      const registration = registrationOption.value;
      const health = yield* probeHealth(registration.url, 500, client);
      if (
        health.version !== options.version ||
        health.id !== registration.id ||
        health.pid !== registration.pid
      ) {
        return yield* new NoDaemon();
      }
      return registration;
    });

    const awaitHealthyRegistration = checkHealthyRegistration.pipe(
      Effect.retry({
        schedule: Schedule.spaced("50 millis"),
        while: (error) => error._tag === "NoDaemon",
      })
    );

    const resultOption = yield* Effect.gen(function* () {
      const first = yield* Effect.race(
        awaitHealthyRegistration.pipe(
          Effect.map(
            (registration) => ({ _tag: "Registered", registration } as const)
          )
        ),
        Effect.tryPromise({
          try: () => proc.exited,
          catch: () => new DaemonCrashed({ exitCode: 1 }),
        }).pipe(
          Effect.catchAll((error) => Effect.succeed(error.exitCode)),
          Effect.map(
            (exitCode) =>
              ({ _tag: "CandidateExited", exitCode: exitCode ?? 1 } as const)
          )
        )
      );
      if (first._tag === "Registered") return first.registration;

      yield* debugLog(
        options.debug,
        "spawned ACN candidate exited while waiting for shared registration",
        {
          pid: proc.pid,
          exitCode: first.exitCode,
        }
      );
      return yield* awaitHealthyRegistration;
    }).pipe(
      Effect.catchTag(
        "NoDaemon",
        () =>
          new DaemonSpawnFailed({
            reason:
              "No compatible ACN became healthy before the startup deadline",
          })
      ),
      Effect.timeoutOption(`${options.timeoutMs} millis`)
    );

    const result = yield* Option.match(resultOption, {
      onNone: () =>
        new DaemonSpawnFailed({
          reason:
            "No compatible ACN became healthy before the startup deadline",
        }),
      onSome: Effect.succeed,
    });

    yield* debugLog(options.debug, "spawned ACN became healthy", {
      url: result.url,
      pid: result.pid,
      id: result.id,
    });
    return result.url;
  });

// ─── Daemon action decision ──────────────────────────────────────────────────

export type DaemonAction =
  | {
      readonly type: "connect";
      readonly url: string;
      readonly reason: "same-version";
    }
  | { readonly type: "spawn"; readonly reason: "missing" | "stale" };

export const decideDaemonAction = (input: {
  readonly registration: Option.Option<AcnRegistration>;
  readonly health: Option.Option<HealthResponse>;
}): DaemonAction => {
  if (Option.isNone(input.registration)) {
    return { type: "spawn", reason: "missing" };
  }

  if (Option.isNone(input.health)) {
    return { type: "spawn", reason: "stale" };
  }

  if (
    input.health.value.version !== input.registration.value.version ||
    input.health.value.id !== input.registration.value.id ||
    input.health.value.pid !== input.registration.value.pid
  ) {
    return { type: "spawn", reason: "stale" };
  }

  const url = input.registration.value.url;
  return { type: "connect", url, reason: "same-version" };
};

// ─── discover helper ─────────────────────────────────────────────────────────

/**
 * Reads the registration file, health-checks the daemon, returns the URL if
 * it's alive and matches the target version.
 */
const discoverUrl = (
  options: {
    readonly dataDir: string;
    readonly version: string;
    readonly probeTimeoutMs: number;
    readonly debug: boolean;
  },
  deps: {
    readonly fs: FileSystem;
    readonly client: HttpClient.HttpClient;
  }
): Effect.Effect<Option.Option<string>, DaemonError, never> =>
  Effect.gen(function* () {
    const regPath = registrationPath(options.dataDir, options.version);
    const registration = yield* readRegistration(regPath, deps.fs);
    yield* debugLog(
      options.debug,
      "read registration",
      Option.match(registration, {
        onNone: () => ({ state: "missing", version: options.version }),
        onSome: (reg) => ({
          state: "present",
          url: reg.url,
          version: reg.version,
          pid: reg.pid,
          id: reg.id,
        }),
      })
    );

    const health = yield* Option.match(registration, {
      onNone: () => Effect.succeed(Option.none<HealthResponse>()),
      onSome: (reg) =>
        probeHealth(reg.url, options.probeTimeoutMs, deps.client).pipe(
          Effect.flatMap((h) =>
            h.version === options.version &&
            h.id === reg.id &&
            h.pid === reg.pid
              ? Effect.succeed(Option.some(h))
              : Effect.succeed(Option.none<HealthResponse>())
          ),
          Effect.catchAll(() => Effect.succeed(Option.none<HealthResponse>()))
        ),
    });
    yield* debugLog(
      options.debug,
      "probed health",
      Option.match(health, {
        onNone: () => ({ state: "unhealthy", version: options.version }),
        onSome: (h) => ({ state: "healthy", version: h.version }),
      })
    );

    const action = decideDaemonAction({ registration, health });
    return action.type === "connect"
      ? Option.some(action.url)
      : Option.none<string>();
  });

// ─── Public factory ──────────────────────────────────────────────────────────

/**
 * Options for the local daemon spawner. These are passed through from
 * `EnsureDaemonOptions` by the caller.
 */
export interface LocalSpawnerOptions {
  readonly binaryPath?: string;
  readonly version?: string;
  readonly spawnTimeoutMs?: number;
  readonly probeTimeoutMs?: number;
  readonly debug?: boolean;
  /** Test/embedding override. Defaults to ~/.magnitude. */
  readonly dataDir?: string;
}

/**
 * Creates a local `DaemonSpawner` that captures `FileSystem`, `HttpClient`,
 * and `CommandExecutor` at construction time.
 *
 * The returned spawner's `discover` and `spawn` methods require only `never`
 * — all dependencies are sealed inside.
 *
 * @param spawnProcess — Bun.spawn or child_process.spawn adapter
 * @param options — version, timeouts, debug, binaryPath (passed from EnsureDaemonOptions)
 */
export const makeLocalDaemonSpawner = (
  spawnProcess: SpawnProcess,
  options: LocalSpawnerOptions = {}
): Effect.Effect<
  DaemonSpawner,
  never,
  FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem;
    const client = yield* HttpClient.HttpClient;
    const cmd = yield* CommandExecutor.CommandExecutor;

    const dataDir = options.dataDir ?? defaultDataDir();
    const targetVersion = options.version ?? SDK_VERSION;
    const debug = options.debug ?? process.env.MAGNITUDE_ACN_DEBUG === "1";
    const spawnTimeoutMs = options.spawnTimeoutMs ?? 10000;
    const probeTimeoutMs = options.probeTimeoutMs ?? 2000;

    const deps = { fs, client };

    return {
      discover: () =>
        discoverUrl(
          { dataDir, version: targetVersion, probeTimeoutMs, debug },
          deps
        ),

      spawn: (command) =>
        withSpawnElection(
          spawnElectionPath(dataDir, targetVersion),
          spawnTimeoutMs,
          fs,
          Effect.gen(function* () {
            // Mandatory post-election recheck: the observation that caused this
            // caller to enter the election may already be stale.
            const existing = yield* discoverUrl(
              { dataDir, version: targetVersion, probeTimeoutMs, debug },
              deps
            );
            if (Option.isSome(existing)) {
              return existing.value;
            }

            const resolvedCommand =
              command ??
              (yield* resolveBinaryCommand({
                binaryPath: options.binaryPath,
                version: targetVersion,
              }).pipe(
                Effect.provideService(FileSystem, fs),
                Effect.provideService(HttpClient.HttpClient, client),
                Effect.provideService(CommandExecutor.CommandExecutor, cmd),
                Effect.map((resolved) =>
                  debug ? [...resolved.command, "--debug"] : resolved.command
                )
              ));

            return yield* spawnDaemon(
              resolvedCommand,
              {
                dataDir,
                version: targetVersion,
                timeoutMs: spawnTimeoutMs,
                debug,
              },
              { fs, client, spawnProcess }
            );
          }),
        ),
    } satisfies DaemonSpawner;
  });
