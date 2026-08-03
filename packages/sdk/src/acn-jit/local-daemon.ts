/** Local daemon discovery and launch for Bun and Node environments. */
import {
  Array as Arr,
  Context,
  Effect,
  Option,
  Ref,
  Schedule,
  Schema,
  Stream,
} from "effect";
import type * as ParseResult from "effect/ParseResult";
import { FileSystem } from "@effect/platform/FileSystem";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import type { PlatformError } from "@effect/platform/Error";
import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import * as Path from "@effect/platform/Path";
import * as NodePath from "node:path";
import { readStructuredFile } from "@magnitudedev/storage";
import {
  AcnVersionRegistrySchema,
  AcnHealthResponseSchema,
  type AcnHealthResponse,
  type AcnInstallationPlan,
  type AcnRegistration,
  type AcnOwnerId,
  type AcnEndpoint,
  type AcnStartupProgress,
} from "@magnitudedev/acn-protocol";
import type { ArtifactInstallationEvent } from "@magnitudedev/release";
import {
  DaemonCrashed,
  DaemonError,
  DaemonSpawnFailed,
  NoDaemon,
  RegistrationFileInvalid,
} from "./errors";
import { resolveBinaryCommand, defaultDataDir } from "../binary";
import { SDK_VERSION } from "../version";
import { canUseDaemonVersion } from "./release-precedence";
import {
  acnLifecycleObservationFromHealthState,
  type AcnLifecycleObservation,
} from "./lifecycle";
import type { DaemonDiscovery, DaemonStatus } from "./daemon-discovery";
import type {
  DaemonLaunchEvent,
  DaemonLauncher,
} from "./daemon-launcher";

type EmitStartupObservation = (
  observation: AcnLifecycleObservation
) => Effect.Effect<void>;

const endpointOf = (daemon: AcnEndpoint): AcnEndpoint => ({
  id: daemon.id,
  version: daemon.version,
  url: daemon.url,
});

/**
 * A host-owned detached process with Effect-native supervision operations.
 */
export interface ChildProcess {
  readonly pid: Option.Option<number>;
  readonly exited: Effect.Effect<number>;
  readonly diagnostic: Effect.Effect<Option.Option<string>>;
  readonly kill: (signal: NodeJS.Signals) => Effect.Effect<void>;
}

export interface ChildProcessSpawner {
  readonly spawn: (
    command: Arr.NonEmptyReadonlyArray<string>
  ) => Effect.Effect<ChildProcess, DaemonSpawnFailed>;
}

export const ChildProcessSpawner = Context.GenericTag<ChildProcessSpawner>(
  "@magnitudedev/sdk/ChildProcessSpawner"
);

// ─── Internal types ──────────────────────────────────────────────────────────

type DebugField =
  | string
  | number
  | boolean
  | null
  | ReadonlyArray<string | number | boolean | null>;

const debugLog = (
  enabled: boolean,
  message: string,
  fields: Readonly<Record<string, DebugField>> = {}
): Effect.Effect<void> =>
  enabled
    ? Effect.logDebug(message).pipe(Effect.annotateLogs(fields))
    : Effect.void;

// ─── Path helpers ────────────────────────────────────────────────────────────

const acnDirectory = (dataDir: string): string => NodePath.join(dataDir, "acn");

const registrationPath = (dataDir: string): string =>
  NodePath.join(acnDirectory(dataDir), "registry.json");

const spawnElectionPath = (dataDir: string): string =>
  NodePath.join(acnDirectory(dataDir), "spawn-election");

const staleSpawnElectionPath = (path: string, token: string): string =>
  `${path}.stale-${encodeURIComponent(token)}`;

interface SpawnElectionClaim {
  readonly path: string;
  readonly token: string;
}

const SpawnElectionOwnerSchema = Schema.Struct({
  token: Schema.String,
  pid: Schema.Number,
});
type SpawnElectionOwner = typeof SpawnElectionOwnerSchema.Type;

const platformErrorReason = (
  cause: PlatformError
): Option.Option<string> =>
  cause._tag === "SystemError"
    ? Option.some(cause.reason)
    : Option.none();

const hasPlatformErrorReason = (
  cause: PlatformError,
  expected: string
): boolean =>
  Option.exists(platformErrorReason(cause), (reason) => reason === expected);

const electionFailure = (
  operation: string,
  cause: PlatformError | ParseResult.ParseError
) =>
  new DaemonSpawnFailed({
    reason: `${operation}: ${String(cause)}`,
  });

const readSpawnElectionOwner = (
  path: string,
  fs: FileSystem
): Effect.Effect<Option.Option<SpawnElectionOwner>> =>
  fs.readFileString(path).pipe(
    Effect.flatMap(
      Schema.decodeUnknown(Schema.parseJson(SpawnElectionOwnerSchema))
    ),
    Effect.map(Option.some),
    Effect.catchAll(() => Effect.succeed(Option.none()))
  );

const SPAWN_ELECTION_STALE_MS = 10_000;
const SPAWN_ELECTION_WAIT_MS = 2_000;

const tryAcquireSpawnElection = (
  path: string,
  staleAfterMs: number,
  fs: FileSystem
): Effect.Effect<Option.Option<SpawnElectionClaim>, DaemonSpawnFailed> =>
  Effect.gen(function* () {
    yield* fs
      .makeDirectory(NodePath.dirname(path), { recursive: true })
      .pipe(
        Effect.mapError((cause) =>
          electionFailure("Failed to create ACN registry directory", cause)
        )
      );
    yield* fs.chmod(NodePath.dirname(path), 0o700).pipe(
      Effect.mapError((cause) =>
        electionFailure("Failed to secure ACN registry directory", cause)
      )
    );
    const claim = {
      path,
      token: crypto.randomUUID(),
    } satisfies SpawnElectionClaim;
    const encodedOwner = yield* Schema.encode(
      Schema.parseJson(SpawnElectionOwnerSchema)
    )({ token: claim.token, pid: process.pid }).pipe(
      Effect.mapError((cause) =>
        electionFailure("Failed to encode ACN spawn election owner", cause)
      )
    );
    const publicationPath = `${path}.publishing-${encodeURIComponent(
      claim.token
    )}`;
    const acquired = yield* fs
      .writeFileString(publicationPath, encodedOwner, { mode: 0o600 })
      .pipe(
        Effect.flatMap(() => fs.link(publicationPath, path)),
        Effect.as(true),
        Effect.catchAll((cause) =>
          hasPlatformErrorReason(cause, "AlreadyExists")
            ? Effect.succeed(false)
            : Effect.fail(
                electionFailure("Failed to acquire ACN spawn election", cause)
              )
        ),
        Effect.ensuring(
          fs.remove(publicationPath, { force: true }).pipe(Effect.ignore)
        )
      );
    if (acquired) {
      return Option.some(claim);
    }

    // Election is an optimization, not a correctness boundary. Exact ACN
    // publication and peer removal resolve any duplicate candidate, so an old
    // claim may be quarantined by age without granting it authority forever.
    const info = yield* fs.stat(path).pipe(
      Effect.map(Option.some),
      Effect.catchAll((cause) =>
        hasPlatformErrorReason(cause, "NotFound")
          ? Effect.succeed(Option.none())
          : Effect.fail(
              electionFailure("Failed to inspect ACN spawn election", cause)
            )
      )
    );
    if (Option.isSome(info)) {
      const isStale = Option.exists(
        info.value.mtime,
        (modifiedAt) => Date.now() - modifiedAt.getTime() > staleAfterMs
      );
      if (isStale) {
        const observedOwner = yield* readSpawnElectionOwner(path, fs);
        if (Option.isSome(observedOwner)) {
          const tombstone = staleSpawnElectionPath(
            path,
            observedOwner.value.token
          );
          // Claim recovery by atomically hard-linking the exact owner record to
          // its retained tombstone. Only one contender can publish that link.
          // A later contender may finish removal if the winner crashed after
          // linking, but the token checks prevent it from touching a new owner.
          const linked = yield* fs.link(path, tombstone).pipe(
            Effect.as(true),
            Effect.catchAll((cause) =>
              hasPlatformErrorReason(cause, "AlreadyExists")
                ? Effect.succeed(false)
                : Effect.fail(
                    electionFailure(
                      "Failed to quarantine stale ACN spawn election",
                      cause
                    )
                  )
            )
          );
          const quarantinedOwner = linked
            ? observedOwner
            : yield* readSpawnElectionOwner(tombstone, fs);
          const currentOwner = yield* readSpawnElectionOwner(path, fs);
          if (
            Option.isSome(quarantinedOwner) &&
            Option.isSome(currentOwner) &&
            quarantinedOwner.value.token === observedOwner.value.token &&
            quarantinedOwner.value.pid === observedOwner.value.pid &&
            currentOwner.value.token === observedOwner.value.token &&
            currentOwner.value.pid === observedOwner.value.pid
          ) {
            yield* fs
              .remove(path)
              .pipe(
                Effect.catchAll((cause) =>
                  hasPlatformErrorReason(cause, "NotFound")
                    ? Effect.void
                    : Effect.fail(
                        electionFailure(
                          "Failed to recover stale ACN spawn election",
                          cause
                        )
                      )
                )
              );
          }
        }
      }
    }
    return Option.none();
  });

const releaseSpawnElection = (
  claim: SpawnElectionClaim,
  fs: FileSystem
): Effect.Effect<void> =>
  readSpawnElectionOwner(claim.path, fs).pipe(
    Effect.flatMap(
      Option.match({
        onNone: () => Effect.void,
        onSome: (owner) =>
          owner.token === claim.token ? fs.remove(claim.path) : Effect.void,
      })
    ),
    Effect.catchAll(() => Effect.void)
  );

const withSpawnElection = <A, E>(
  path: string,
  fs: FileSystem,
  emitObservation: EmitStartupObservation,
  effect: Effect.Effect<A, E, never>
): Effect.Effect<A, E | DaemonSpawnFailed, never> => {
  const acquire = tryAcquireSpawnElection(
    path,
    SPAWN_ELECTION_STALE_MS,
    fs
  ).pipe(
    Effect.tap((claim) =>
      Option.isNone(claim)
        ? emitObservation({ _tag: "Starting", phase: "WaitingForOwner" })
        : Effect.void
    ),
    Effect.filterOrFail(Option.isSome, () => new NoDaemon()),
    Effect.map((claim) => claim.value),
    Effect.retry({
      schedule: Schedule.spaced("50 millis"),
      while: (failure) => failure._tag === "NoDaemon",
    }),
    Effect.mapError((failure) =>
      failure._tag === "NoDaemon"
        ? new DaemonSpawnFailed({
            reason: "ACN spawn election ended unexpectedly",
          })
        : failure
    )
  );
  return acquire.pipe(
    Effect.timeoutOption(`${SPAWN_ELECTION_WAIT_MS} millis`),
    Effect.flatMap(
      Option.match({
        onNone: () =>
          Effect.logWarning(
            "ACN spawn election did not converge; continuing with exact ACN reconciliation",
          ).pipe(Effect.zipRight(effect)),
        onSome: (claim) =>
          effect.pipe(Effect.ensuring(releaseSpawnElection(claim, fs))),
      }),
    ),
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
    return result.value.registration;
  });

// ─── Health probing ──────────────────────────────────────────────────────────

const probeHealth = (
  url: string,
  timeoutMs: number,
  client: HttpClient.HttpClient
): Effect.Effect<AcnHealthResponse, NoDaemon, never> =>
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
    const health = yield* Schema.decodeUnknown(AcnHealthResponseSchema)(
      json
    ).pipe(Effect.mapError(() => new NoDaemon()));

    if (health.service !== "magnitude-acn") {
      return yield* new NoDaemon();
    }

    return health;
  });

const reportHealthState = (
  state: AcnHealthResponse["state"],
  emitObservation: EmitStartupObservation
): Effect.Effect<void> =>
  Option.match(acnLifecycleObservationFromHealthState(state), {
    onNone: () => Effect.void,
    onSome: emitObservation,
  });

/** Reads and identity-validates the canonical registration and health state. */
const readCurrentDaemon = (
  options: {
    readonly dataDir: string;
    readonly probeTimeoutMs: number;
    readonly debug: boolean;
  },
  deps: {
    readonly fs: FileSystem;
    readonly client: HttpClient.HttpClient;
  }
): Effect.Effect<Option.Option<DaemonStatus>, RegistrationFileInvalid, never> =>
  Effect.gen(function* () {
    const registration = yield* readRegistration(
      registrationPath(options.dataDir),
      deps.fs,
    );
    if (Option.isNone(registration)) return Option.none();
    const health = yield* probeHealth(
      registration.value.url,
      options.probeTimeoutMs,
      deps.client,
    ).pipe(
      Effect.map(Option.some),
      Effect.catchAll(() => Effect.succeed(Option.none<AcnHealthResponse>())),
    );
    if (
      Option.isNone(health) ||
      health.value.version !== registration.value.version ||
      health.value.id !== registration.value.id ||
      health.value.pid !== registration.value.pid
    ) {
      yield* debugLog(options.debug, "canonical ACN is unreachable or stale");
      return Option.none();
    }
    return Option.some({
      ...endpointOf(registration.value),
      pid: registration.value.pid,
      state: health.value.state,
    });
  });

const daemonDownloadObservation = (
  plan: AcnInstallationPlan,
  event: Extract<ArtifactInstallationEvent, { readonly _tag: "Downloading" }>
): AcnLifecycleObservation => ({
  _tag: "Installing",
  phase: "DownloadingDaemon",
  plan,
  progress: Option.some({
    completed: event.progress.acceptedBytes,
    totalBytes: event.progress.totalBytes,
    unit: "Bytes",
    attempt: Option.some(event.progress.attempt),
  }),
});

// ─── Spawn daemon (wait-for-registration pipeline) ───────────────────────────

const spawnDaemon = (
  command: Arr.NonEmptyReadonlyArray<string>,
  options: {
    readonly dataDir: string;
    readonly version: string;
    readonly publicationTimeoutMs: number;
    readonly initialOwnerId: Option.Option<AcnOwnerId>;
    readonly debug: boolean;
    readonly emitObservation: EmitStartupObservation;
  },
  deps: {
    readonly fs: FileSystem;
    readonly client: HttpClient.HttpClient;
    readonly childProcessSpawner: ChildProcessSpawner;
  }
): Effect.Effect<
  {
    readonly ready: Effect.Effect<
      AcnEndpoint,
      DaemonSpawnFailed | DaemonCrashed | RegistrationFileInvalid,
      never
    >;
  },
  DaemonSpawnFailed | DaemonCrashed | RegistrationFileInvalid,
  never
> =>
  Effect.gen(function* () {
    const { fs, client, childProcessSpawner } = deps;
    yield* debugLog(options.debug, "spawning ACN", {
      command: command.join(" "),
      detached: true,
    });

    const proc = yield* childProcessSpawner.spawn(command);

    yield* debugLog(options.debug, "ACN process spawned", {
      pid: Option.getOrNull(proc.pid),
    });

    const checkPublishedRegistration: Effect.Effect<
      DaemonStatus,
      NoDaemon | RegistrationFileInvalid | DaemonSpawnFailed,
      never
    > = Effect.gen(function* () {
      const observed = yield* readCurrentDaemon(
        {
          dataDir: options.dataDir,
          probeTimeoutMs: 500,
          debug: options.debug,
        },
        { fs, client },
      );
      if (Option.isNone(observed)) return yield* new NoDaemon();
      const status = observed.value;
      if (Option.contains(options.initialOwnerId, status.id)) {
        return yield* new NoDaemon();
      }
      if (!canUseDaemonVersion(options.version, status.version)) {
        return yield* new NoDaemon();
      }
      yield* reportHealthState(status.state, options.emitObservation);
      if (status.state._tag === "Stopping") {
        const stopping = status.state;
        return yield* new DaemonSpawnFailed({
          reason: Option.getOrElse(
            stopping.safeDetail,
            () => `ACN is stopping (${stopping.reason})`,
          ),
        });
      }
      return status;
    });

    const awaitPublishedRegistration = checkPublishedRegistration.pipe(
      Effect.retry({
        schedule: Schedule.spaced("50 millis"),
        while: (error) => error._tag === "NoDaemon",
      }),
      Effect.catchTag("NoDaemon", () =>
        Effect.fail(
          new DaemonSpawnFailed({
            reason: "ACN registration observation ended unexpectedly",
          })
        )
      )
    );

    const candidateExit = proc.exited.pipe(
      Effect.flatMap((exitCode) =>
        proc.diagnostic.pipe(
          Effect.flatMap((diagnostic) =>
            Effect.fail(
              new DaemonCrashed({
                exitCode,
                diagnostic,
              })
            )
          )
        )
      )
    );

    const publication = yield* Effect.raceFirst(
      awaitPublishedRegistration,
      candidateExit
    ).pipe(
      Effect.catchTag("DaemonCrashed", (candidateFailure) =>
        checkPublishedRegistration.pipe(Effect.mapError(() => candidateFailure))
      ),
      Effect.timeoutOption(`${options.publicationTimeoutMs} millis`)
    );
    const published = yield* Option.match(publication, {
      onSome: Effect.succeed,
      onNone: () =>
        Effect.gen(function* () {
          yield* proc.kill("SIGTERM");
          const terminated = yield* proc.exited.pipe(
            Effect.timeoutOption("1 second")
          );
          if (Option.isNone(terminated)) {
            yield* proc.kill("SIGKILL");
            yield* proc.exited.pipe(Effect.timeout("1 second"), Effect.ignore);
          }
          const diagnostic = yield* proc.diagnostic;
          const publicationFailure = new DaemonSpawnFailed({
            reason: Option.match(diagnostic, {
              onNone: () =>
                "ACN did not publish its startup endpoint before the deadline",
              onSome: (detail) =>
                `ACN did not publish its startup endpoint before the deadline: ${detail}`,
            }),
          });
          return yield* checkPublishedRegistration.pipe(
            Effect.mapError(() => publicationFailure)
          );
        }),
    });
    const publicationBelongsToCandidate = Option.exists(
      proc.pid,
      (pid) => pid === published.pid
    );

    const awaitReady = checkPublishedRegistration.pipe(
      Effect.filterOrFail(
        ({ state }) => state._tag === "Ready",
        () => new NoDaemon()
      ),
      Effect.retry({
        schedule: Schedule.spaced("100 millis"),
        while: (error) => error._tag === "NoDaemon",
      }),
      Effect.catchTag("NoDaemon", () =>
        Effect.fail(
          new DaemonSpawnFailed({
            reason: "ACN readiness observation ended unexpectedly",
          })
        )
      )
    );

    return {
      ready: (publicationBelongsToCandidate
        ? Effect.raceFirst(awaitReady, candidateExit)
        : awaitReady
      ).pipe(
        Effect.tap((result) =>
          debugLog(options.debug, "spawned ACN became ready", {
            url: result.url,
            pid: result.pid,
            id: result.id,
          })
        ),
        Effect.map(endpointOf)
      ),
    };
  });

// ─── Public factories ───────────────────────────────────────────────────────

/**
 * Host settings required to observe the canonical local daemon.
 */
export interface LocalDaemonDiscoveryOptions {
  readonly probeTimeoutMs?: number;
  readonly debug?: boolean;
  /** Test/embedding override. Defaults to ~/.magnitude. */
  readonly dataDir?: string;
}

/** Host settings required to launch and observe a local daemon candidate. */
export interface LocalDaemonLauncherOptions extends LocalDaemonDiscoveryOptions {
  readonly binaryPath?: string;
  readonly version?: string;
  /** Deadline from candidate spawn until its early registration is observable. */
  readonly publicationTimeoutMs?: number;
}

/** Creates local daemon discovery from canonical registration and health. */
export const makeLocalDaemonDiscovery = (
  options: LocalDaemonDiscoveryOptions = {}
): Effect.Effect<DaemonDiscovery, never, FileSystem | HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem;
    const client = yield* HttpClient.HttpClient;
    const dataDir = options.dataDir ?? defaultDataDir();
    const probeTimeoutMs = options.probeTimeoutMs ?? 2_000;
    const debug = options.debug ?? process.env.MAGNITUDE_ACN_DEBUG === "1";
    return {
      current: () =>
        readCurrentDaemon(
          { dataDir, probeTimeoutMs, debug },
          { fs, client },
        ),
    } satisfies DaemonDiscovery;
  });

/** Creates the launch mutation and captures its host-specific dependencies. */
export const makeLocalDaemonLauncher = (
  options: LocalDaemonLauncherOptions = {}
): Effect.Effect<
  DaemonLauncher,
  never,
  | FileSystem
  | HttpClient.HttpClient
  | CommandExecutor.CommandExecutor
  | Path.Path
  | ChildProcessSpawner
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem;
    const client = yield* HttpClient.HttpClient;
    const cmd = yield* CommandExecutor.CommandExecutor;
    const path = yield* Path.Path;
    const childProcessSpawner = yield* ChildProcessSpawner;

    const dataDir = options.dataDir ?? defaultDataDir();
    const targetVersion = options.version ?? SDK_VERSION;
    const debug = options.debug ?? process.env.MAGNITUDE_ACN_DEBUG === "1";
    const publicationTimeoutMs = options.publicationTimeoutMs ?? 10_000;
    const probeTimeoutMs = options.probeTimeoutMs ?? 2000;

    const deps = { fs, client };
    const observeCurrent = readCurrentDaemon(
      { dataDir, probeTimeoutMs, debug },
      deps,
    );
    const canJoinLaunch = (status: DaemonStatus): boolean =>
      status.state._tag !== "Stopping" &&
      canUseDaemonVersion(targetVersion, status.version);

    return {
      launch: (command) =>
        Stream.asyncPush<DaemonLaunchEvent, DaemonError>(
          (emit) => {
            const emitObservation: EmitStartupObservation = (observation) =>
              Effect.sync(() => {
                emit.single({ _tag: "Observation", observation });
              });
            const startup = Effect.gen(function* () {
              const initialOwnerId = yield* readRegistration(
                registrationPath(dataDir),
                fs,
              ).pipe(Effect.map(Option.map((registration) => registration.id)));
              const isConcurrentLaunch = (id: string): boolean =>
                !Option.contains(initialOwnerId, id);

              let launchCandidate!: Effect.Effect<
                { readonly ready: Effect.Effect<AcnEndpoint, DaemonError> },
                DaemonError
              >;
              const awaitConcurrentLaunch: Effect.Effect<AcnEndpoint, DaemonError> =
                Effect.suspend(() =>
                  observeCurrent.pipe(
                    Effect.flatMap(
                      Option.match({
                        onNone: () =>
                          launchCandidate.pipe(Effect.flatMap(({ ready }) => ready)),
                        onSome: (status) => {
                          if (!isConcurrentLaunch(status.id) || !canJoinLaunch(status)) {
                            return launchCandidate.pipe(
                              Effect.flatMap(({ ready }) => ready),
                            );
                          }
                          if (status.state._tag === "Ready") {
                            return Effect.succeed(endpointOf(status));
                          }
                          return reportHealthState(status.state, emitObservation).pipe(
                            Effect.zipRight(Effect.sleep("100 millis")),
                            Effect.zipRight(awaitConcurrentLaunch),
                          );
                        },
                      }),
                    ),
                  ),
                );
              launchCandidate = withSpawnElection(
                spawnElectionPath(dataDir),
                fs,
                emitObservation,
                Effect.gen(function* () {
                  // Mandatory post-election convergence. A contender never
                  // acts on the stale state that caused it to enter election.
                  const current = yield* observeCurrent;
                  if (
                    Option.isSome(current) &&
                    isConcurrentLaunch(current.value.id) &&
                    canJoinLaunch(current.value)
                  ) {
                    if (current.value.state._tag === "Ready") {
                      return { ready: Effect.succeed(endpointOf(current.value)) };
                    }
                    yield* reportHealthState(current.value.state, emitObservation);
                    return { ready: awaitConcurrentLaunch };
                  }

                  const acquisitionPlan = yield* Ref.make<
                    Option.Option<AcnInstallationPlan>
                  >(Option.none());
                  const resolvedCommand = yield* Option.match(command, {
                    onSome: (value) => Effect.succeed(Array.from(value)),
                    onNone: () =>
                      resolveBinaryCommand({
                        binaryPath: options.binaryPath,
                        version: targetVersion,
                        dataDir,
                        acquisitionObserver: Option.some({
                          report: (event) =>
                            Effect.gen(function* () {
                              switch (event._tag) {
                                case "Planned":
                                  return yield* Ref.set(
                                    acquisitionPlan,
                                    Option.some(event.plan)
                                  );
                                case "Artifact": {
                                  if (event.event._tag !== "Downloading") return;
                                  const plan = yield* Ref.get(acquisitionPlan);
                                  if (Option.isNone(plan)) {
                                    return yield* Effect.dieMessage(
                                      "ACN artifact download began before its installation plan"
                                    );
                                  }
                                  return yield* emitObservation(
                                    daemonDownloadObservation(
                                      plan.value,
                                      event.event
                                    )
                                  );
                                }
                              }
                            }),
                        }),
                      }).pipe(
                        Effect.provideService(FileSystem, fs),
                        Effect.provideService(HttpClient.HttpClient, client),
                        Effect.provideService(
                          CommandExecutor.CommandExecutor,
                          cmd
                        ),
                        Effect.provideService(Path.Path, path),
                        Effect.map((resolved) =>
                          debug
                            ? [...resolved.command, "--debug"]
                            : resolved.command
                        )
                      ),
                  });
                  if (!Arr.isNonEmptyReadonlyArray(resolvedCommand)) {
                    return yield* new DaemonSpawnFailed({
                      reason: "Cannot spawn an empty Magnitude command",
                    });
                  }

                  yield* emitObservation({
                    _tag: "Starting",
                    phase: "LaunchingAcn",
                  });
                  return yield* spawnDaemon(
                    resolvedCommand,
                    {
                      dataDir,
                      version: targetVersion,
                      publicationTimeoutMs,
                      initialOwnerId,
                      debug,
                      emitObservation,
                    },
                    { fs, client, childProcessSpawner }
                  );
                })
              );
              const awaitReady = yield* launchCandidate;
              return yield* awaitReady.ready;
            });
            return startup.pipe(
              Effect.match({
                onFailure: (error) => {
                  emit.fail(error);
                },
                onSuccess: (endpoint) => {
                  emit.single({ _tag: "Ready", endpoint });
                  emit.end();
                },
              }),
              Effect.forkScoped
            );
          },
          { bufferSize: "unbounded" }
        ),
    } satisfies DaemonLauncher;
  });
