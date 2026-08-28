import * as Command from "@effect/platform/Command";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import * as FileSystem from "@effect/platform/FileSystem";
import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientError from "@effect/platform/HttpClientError";
import * as Path from "@effect/platform/Path";
import {
  IcnBinaryIdentity,
  IcnStartupProgressRecord,
  IcnStartupRecord,
} from "@magnitudedev/icn-protocol";
import { GeneratedClientTransportError } from "@magnitudedev/openapi-effect/client-runtime";
import { FSM } from "@magnitudedev/utils";
import { dirname, join } from "node:path";
import {
  Context,
  Deferred,
  Duration,
  Effect,
  Layer,
  Option,
  Random,
  Ref,
  Schedule,
  Schema,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect";
import { installationLoaderEnvironment } from "./installation-environment.js";
import { ICN_EXECUTABLE_NAME } from "@magnitudedev/release/executables";
import {
  IcnApiIncompatible,
  IcnBinaryNotExecutable,
  IcnBinaryNotFound,
  IcnCapabilityMissing,
  IcnExitedBeforeReady,
  IcnHealthIdentityMismatch,
  IcnIdentityProbeTimedOut,
  IcnNativeBuildIncompatible,
  IcnReadinessCommitRejected,
  IcnReadinessTimedOut,
  IcnShutdownTimedOut,
  IcnStartupIdentityMismatch,
  IcnStartupOriginInvalid,
  IcnStartupOriginNotLoopback,
  IcnStartupRecordTimedOut,
  IcnTargetIncompatible,
  IcnUnexpectedExit,
  type IcnBinaryResolutionError,
  type IcnLifecycleError,
} from "./errors.js";
import {
  makeIcnApiClient,
} from "@magnitudedev/icn-protocol/client";
import { resolveReleaseIcnInstallation } from "./release-installation.js";
import {
  IcnPreparationReporter,
  icnPreparationBackend,
  type IcnPreparationReporter as IcnPreparationReporterService,
} from "./preparation.js";

export * from "./preparation.js";
export * from "./errors.js";

const PositiveInt = Schema.Int.pipe(Schema.greaterThan(0));
const NonEmpty = Schema.String.pipe(Schema.minLength(1));

export const IcnBinarySource = Schema.Union(
  Schema.TaggedStruct("Installation", {
    path: NonEmpty,
  }),
  Schema.TaggedStruct("Release", {
    version: NonEmpty,
    dataDir: NonEmpty,
    releaseBaseUrl: NonEmpty,
  })
);
export type IcnBinarySource = typeof IcnBinarySource.Type;

export class IcnBinaryResolutionConfig extends Schema.Class<IcnBinaryResolutionConfig>(
  "IcnBinaryResolutionConfig"
)({
  source: IcnBinarySource,
  supportedApiVersion: PositiveInt,
  expectedNativeBuild: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
  expectedTarget: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
  requiredCapabilities: Schema.Array(NonEmpty),
  probeTimeout: Schema.DurationFromSelf.pipe(
    Schema.greaterThanDuration(Duration.zero)
  ),
}) {}

export class IcnStorageConfig extends Schema.Class<IcnStorageConfig>(
  "IcnStorageConfig"
)({
  modelStore: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
  cacheRoot: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
  huggingFaceCaches: Schema.Array(NonEmpty),
}) {}

export class IcnLifecycleConfig extends Schema.Class<IcnLifecycleConfig>(
  "IcnLifecycleConfig"
)({
  binary: IcnBinaryResolutionConfig,
  storage: IcnStorageConfig,
  host: Schema.Literal("127.0.0.1", "::1"),
  startupTimeout: Schema.DurationFromSelf.pipe(
    Schema.greaterThanDuration(Duration.zero)
  ),
  gracefulShutdownTimeout: Schema.DurationFromSelf.pipe(
    Schema.greaterThanDuration(Duration.zero)
  ),
  forceShutdownTimeout: Schema.DurationFromSelf.pipe(
    Schema.greaterThanDuration(Duration.zero)
  ),
  outputLimitBytes: PositiveInt,
}) {}

export interface ResolvedIcnBinary {
  readonly path: string;
  readonly identity: IcnBinaryIdentity;
  readonly installation: string;
  readonly environment: Readonly<Record<string, string>>;
}

const resolveCandidate = (
  source: IcnBinarySource,
) =>
  Effect.gen(function* () {
    const reporter = yield* IcnPreparationReporter;
    yield* reporter.report({ _tag: "Resolving" });
    if (source._tag === "Installation") {
      const root = dirname(source.path);
      return {
        path: join(
          root,
          "bin",
          `${ICN_EXECUTABLE_NAME}${process.platform === "win32" ? ".exe" : ""}`,
        ),
        installation: source.path,
        environment: installationLoaderEnvironment(join(root, "runtime")),
      };
    }
    const installation = yield* resolveReleaseIcnInstallation(
      source.version,
      source.dataDir,
      source.releaseBaseUrl,
    );
    return {
      path: installation.binaryPath,
      installation: installation.declarationPath,
      environment: installation.environment,
    };
  });

export interface IcnBinaryResolverService {
  readonly resolve: (
    config: IcnBinaryResolutionConfig
  ) => Effect.Effect<ResolvedIcnBinary, IcnBinaryResolutionError>;
}

export class IcnBinaryResolver extends Context.Tag(
  "@magnitudedev/icn/IcnBinaryResolver"
)<IcnBinaryResolver, IcnBinaryResolverService>() {}

export const makeIcnBinaryResolver = () => Layer.effect(
  IcnBinaryResolver,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const executor = yield* CommandExecutor.CommandExecutor;
    const path = yield* Path.Path;
    const http = yield* HttpClient.HttpClient;
    const preparation = yield* IcnPreparationReporter;
    return IcnBinaryResolver.of({
      resolve: (config) =>
        Effect.suspend(() =>
          Effect.gen(function* () {
            const candidate = yield* resolveCandidate(config.source).pipe(
              Effect.provideService(IcnPreparationReporter, preparation),
            );
            const exists = yield* fs.exists(candidate.path);
            if (!exists)
              return yield* new IcnBinaryNotFound({ path: candidate.path });
            const canonical = yield* fs.realPath(candidate.path);
            const info = yield* fs.stat(canonical);
            if (
              info.type !== "File" ||
              (!canonical.toLowerCase().endsWith(".exe") &&
                (info.mode & 0o111) === 0)
            )
              return yield* new IcnBinaryNotExecutable({ path: canonical });
            const output = yield* Command.string(
              Command.make(canonical, "version", "--json").pipe(
                Command.env(candidate.environment)
              )
            ).pipe(
              Effect.provideService(CommandExecutor.CommandExecutor, executor),
              Effect.timeoutFail({
                duration: config.probeTimeout,
                onTimeout: () => new IcnIdentityProbeTimedOut({
                  path: canonical,
                  timeout: config.probeTimeout,
                }),
              }),
            );
            const identity = yield* Schema.decodeUnknown(
              Schema.parseJson(IcnBinaryIdentity)
            )(output);
            if (identity.api_version !== config.supportedApiVersion)
              return yield* new IcnApiIncompatible({
                path: canonical,
                expected: config.supportedApiVersion,
                actual: identity.api_version,
              });
            if (
              Option.isSome(config.expectedNativeBuild) &&
              identity.native_build !== config.expectedNativeBuild.value
            )
              return yield* new IcnNativeBuildIncompatible({
                path: canonical,
                expected: config.expectedNativeBuild.value,
                actual: identity.native_build,
              });
            if (
              Option.isSome(config.expectedTarget) &&
              identity.target !== config.expectedTarget.value
            )
              return yield* new IcnTargetIncompatible({
                path: canonical,
                expected: config.expectedTarget.value,
                actual: identity.target,
              });
            const missing = config.requiredCapabilities.find(
              (capability) => !identity.capabilities.includes(capability)
            );
            const missingCapability = Option.fromNullable(missing);
            if (Option.isSome(missingCapability))
              return yield* new IcnCapabilityMissing({
                path: canonical,
                capability: missingCapability.value,
              });
            return {
              path: canonical,
              identity,
              installation: candidate.installation,
              environment: candidate.environment,
            };
          }).pipe(
            Effect.provideService(FileSystem.FileSystem, fs),
            Effect.provideService(CommandExecutor.CommandExecutor, executor),
            Effect.provideService(Path.Path, path),
            Effect.provideService(HttpClient.HttpClient, http),
          )
        ),
    });
  })
);

export interface IcnExit {
  readonly code: number;
  readonly diagnostic: string;
}

export class IcnProcessStarting extends Schema.TaggedClass<IcnProcessStarting>()(
  "Starting",
  {},
) {}

export class IcnProcessReady extends Schema.TaggedClass<IcnProcessReady>()(
  "Ready",
  {},
) {}

export class IcnProcessStopping extends Schema.TaggedClass<IcnProcessStopping>()(
  "Stopping",
  {},
) {}

export class IcnProcessExited extends Schema.TaggedClass<IcnProcessExited>()(
  "Exited",
  {
    code: Schema.Number,
    expected: Schema.Boolean,
  },
) {}

export const IcnProcessLifecycleFsm = FSM.defineFSM(
  {
    Starting: IcnProcessStarting,
    Ready: IcnProcessReady,
    Stopping: IcnProcessStopping,
    Exited: IcnProcessExited,
  },
  {
    Starting: ["Ready", "Stopping", "Exited"],
    Ready: ["Stopping", "Exited"],
    Stopping: ["Exited"],
    Exited: [],
  } as const,
)

export const IcnProcessLifecycleState = Schema.Union(
  IcnProcessStarting,
  IcnProcessReady,
  IcnProcessStopping,
  IcnProcessExited,
)
export type IcnProcessLifecycleState = typeof IcnProcessLifecycleState.Type

export interface IcnProcessService {
  readonly pid: number;
  readonly origin: URL;
  readonly clientOptions: Parameters<typeof makeIcnApiClient>[0];
  readonly instanceId: string;
  readonly binary: ResolvedIcnBinary;
  readonly startup: IcnStartupRecord;
  readonly diagnosticTail: Effect.Effect<string>;
  readonly lifecycle: Effect.Effect<IcnProcessLifecycleState>;
  readonly lifecycleChanges: Stream.Stream<IcnProcessLifecycleState>;
  readonly exit: Effect.Effect<IcnExit, IcnLifecycleError>;
  readonly unexpectedExit: Effect.Effect<never, IcnLifecycleError>;
  readonly shutdown: Effect.Effect<void, IcnLifecycleError>;
}

export class IcnProcess extends Context.Tag("@magnitudedev/icn/IcnProcess")<
  IcnProcess,
  IcnProcessService
>() {}

const appendBounded = (ref: Ref.Ref<string>, chunk: string, limit: number) =>
  Ref.update(ref, (current) => {
    const bytes = new TextEncoder().encode(`${current}${chunk}`);
    if (bytes.byteLength <= limit) return `${current}${chunk}`;
    let start = bytes.byteLength - limit;
    while (start < bytes.byteLength && (bytes[start]! & 0xc0) === 0x80)
      start += 1;
    return new TextDecoder().decode(bytes.subarray(start));
  });

const opaqueInstanceId = Effect.gen(function* () {
  const parts: Array<string> = [];
  for (let index = 0; index < 4; index++)
    parts.push(
      (yield* Random.nextIntBetween(0, 0x1_0000_0000))
        .toString(16)
        .padStart(8, "0")
    );
  return parts.join("");
});

export const renderIcnArguments = (
  config: IcnLifecycleConfig,
  instanceId: string,
  path: string,
): ReadonlyArray<string> => [
  "serve",
  "--bind",
  `${config.host === "::1" ? "[::1]" : config.host}:0`,
  "--instance-id",
  instanceId,
  "--exit-on-stdin-eof",
  "--installation",
  path,
  ...Option.match(config.storage.modelStore, {
    onNone: () => [],
    onSome: (value) => ["--model-store", value],
  }),
  ...Option.match(config.storage.cacheRoot, {
    onNone: () => [],
    onSome: (value) => ["--cache-root", value],
  }),
  ...config.storage.huggingFaceCaches.flatMap((value) => ["--hf-cache", value]),
];

const acquireIcn = (input: IcnLifecycleConfig) =>
  Effect.gen(function* () {
    const config = yield* Schema.validate(IcnLifecycleConfig)(input);
    const resolver = yield* IcnBinaryResolver;
    const reporter = yield* IcnPreparationReporter;
    const binary = yield* resolver.resolve(config.binary);
    yield* reporter.report({ _tag: "Starting" });
    const instanceId = yield* opaqueInstanceId;
    const authorization = yield* opaqueInstanceId;
    const lifecycle = yield* SubscriptionRef.make<IcnProcessLifecycleState>(
      new IcnProcessStarting({}),
    );
    const lifecycleLock = yield* Effect.makeSemaphore(1);
    const shutdownCompletion = yield* Deferred.make<void, IcnLifecycleError>();
    const { process, terminateProcess } = yield* Effect.uninterruptibleMask(() =>
      Effect.gen(function* () {
        const process = yield* Command.start(
          Command.make(
            binary.path,
            ...renderIcnArguments(
              config,
              instanceId,
              binary.installation,
            )
          ).pipe(
            Command.env({
              ...binary.environment,
              MAGNITUDE_ICN_AUTH_TOKEN: authorization,
              HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
            }),
            Command.stdin(Stream.never),
          )
        );
        const waitForProcessExit = process.exitCode;
        const isProcessRunning = process.isRunning;
        const stopAndProve = Effect.gen(function* () {
          if (!(yield* isProcessRunning)) return;
          yield* process.kill("SIGTERM");
          const graceful = yield* waitForProcessExit.pipe(
            Effect.timeoutOption(config.gracefulShutdownTimeout),
          );
          if (Option.isSome(graceful)) return;
          if (yield* isProcessRunning) {
            yield* process.kill("SIGKILL");
          }
          yield* waitForProcessExit.pipe(
            Effect.timeoutFail({
              duration: config.forceShutdownTimeout,
              onTimeout: () => new IcnShutdownTimedOut({
                pid: Number(process.pid),
                timeout: config.forceShutdownTimeout,
              }),
            }),
          );
        });
        const terminateProcess = yield* Effect.cached(stopAndProve);
        yield* Effect.addFinalizer(() =>
          lifecycleLock.withPermits(1)(
            SubscriptionRef.update(lifecycle, (current) =>
              current._tag === "Starting" || current._tag === "Ready"
                ? IcnProcessLifecycleFsm.transition(current, "Stopping", {})
                : current,
            ),
          ).pipe(
            Effect.zipRight(terminateProcess),
            Effect.ignore,
          ),
        );
        return { process, terminateProcess } as const;
      })
    );
    const waitForProcessExit = process.exitCode;
    const output = yield* Ref.make("");
    const startupRecord = yield* Deferred.make<
      IcnStartupRecord,
      IcnLifecycleError
    >();
    const exited = yield* Deferred.make<IcnExit, IcnLifecycleError>();

    yield* process.stdout.pipe(
      Stream.decodeText(),
      Stream.splitLines,
      Stream.runForEach((line) =>
        Effect.gen(function* () {
          yield* appendBounded(output, `${line}\n`, config.outputLimitBytes);
          if (line.startsWith("MAGNITUDE_ICN_PROGRESS ")) {
            const encoded = line.slice("MAGNITUDE_ICN_PROGRESS ".length);
            const record = yield* Schema.decodeUnknown(
              Schema.parseJson(IcnStartupProgressRecord)
            )(encoded);
            yield* reporter.report({
              _tag: "PreparingBackend",
              backend: icnPreparationBackend(record.backend),
            });
            return;
          }
          if (!line.startsWith("MAGNITUDE_ICN_READY ")) return;
          const encoded = line.slice("MAGNITUDE_ICN_READY ".length);
          const record = yield* Schema.decodeUnknown(
            Schema.parseJson(IcnStartupRecord)
          )(encoded);
          yield* Deferred.complete(startupRecord, Effect.succeed(record));
        }).pipe(Effect.catchAll((error) => Deferred.fail(startupRecord, error)))
      ),
      Effect.catchAll((error) => Deferred.fail(startupRecord, error)),
      Effect.forkScoped
    );
    yield* process.stderr.pipe(
      Stream.decodeText(),
      Stream.runForEach((chunk) =>
        appendBounded(output, chunk, config.outputLimitBytes)
      ),
      Effect.option,
      Effect.asVoid,
      Effect.forkScoped
    );
    yield* process.exitCode.pipe(
      Effect.map(Number),
      Effect.flatMap((code) =>
        Ref.get(output).pipe(
          Effect.flatMap((diagnostic) =>
            lifecycleLock.withPermits(1)(Effect.gen(function* () {
              const current = yield* SubscriptionRef.get(lifecycle)
              if (current._tag !== "Exited") {
                yield* SubscriptionRef.set(
                  lifecycle,
                  IcnProcessLifecycleFsm.transition(current, "Exited", {
                    code,
                    expected: current._tag === "Stopping",
                  }),
                )
              }
              yield* Deferred.succeed(exited, { code, diagnostic })
            })),
          )
        )
      ),
      Effect.catchAll((error) => Deferred.fail(exited, error)),
      Effect.forkScoped
    );

    const earlyExit = Deferred.await(exited).pipe(
      Effect.flatMap(({ code, diagnostic }) => Effect.fail(new IcnExitedBeforeReady({
        pid: Number(process.pid),
        code,
        output: diagnostic,
      })))
    );
    const startupResult = yield* Effect.raceFirst(
      Deferred.await(startupRecord),
      earlyExit
    ).pipe(Effect.timeoutOption(config.startupTimeout));
    const startup = yield* Option.match(startupResult, {
      onNone: () => Ref.get(output).pipe(
        Effect.flatMap((currentOutput) => Effect.fail(new IcnStartupRecordTimedOut({
          pid: Number(process.pid),
          timeout: config.startupTimeout,
          output: currentOutput,
        }))),
      ),
      onSome: Effect.succeed,
    });
    if (
      startup.instanceId !== instanceId ||
      startup.pid !== Number(process.pid) ||
      startup.apiVersion !== binary.identity.api_version ||
      startup.nativeBuild !== binary.identity.native_build
    ) {
      const currentOutput = yield* Ref.get(output);
      return yield* new IcnStartupIdentityMismatch({
        pid: Number(process.pid),
        expectedInstanceId: instanceId,
        expectedApiVersion: binary.identity.api_version,
        expectedNativeBuild: binary.identity.native_build,
        actual: startup,
        output: currentOutput,
      });
    }
    const startupOutput = yield* Ref.get(output);
    const origin = yield* Effect.try({
      try: () => new URL(startup.origin),
      catch: () => new IcnStartupOriginInvalid({
        origin: startup.origin,
        output: startupOutput,
      }),
    });
    if (
      (origin.hostname !== "127.0.0.1" &&
        origin.hostname !== "[::1]" &&
        origin.hostname !== "::1") ||
      origin.protocol !== "http:"
    )
      return yield* new IcnStartupOriginNotLoopback({
        origin: startup.origin,
        output: startupOutput,
      });
    const client = yield* makeIcnApiClient({
      baseUrl: origin,
      headers: { authorization: `Bearer ${authorization}` },
    });
    const healthResult = yield* client.system.health({}).pipe(
      Effect.retry({
        schedule: Schedule.spaced("50 millis"),
        while: (cause) =>
          cause instanceof GeneratedClientTransportError &&
          cause.cause instanceof HttpClientError.RequestError,
      }),
      Effect.timeoutOption(config.startupTimeout),
    );
    const health = yield* Option.match(healthResult, {
      onNone: () => Ref.get(output).pipe(
        Effect.flatMap((currentOutput) => Effect.fail(new IcnReadinessTimedOut({
          pid: Number(process.pid),
          timeout: config.startupTimeout,
          output: currentOutput,
        }))),
      ),
      onSome: Effect.succeed,
    });
    if (!health.ready ||
      health.instanceId !== instanceId ||
      health.apiVersion !== binary.identity.api_version ||
      health.nativeBuild !== binary.identity.native_build) {
      const currentOutput = yield* Ref.get(output);
      return yield* new IcnHealthIdentityMismatch({
        expectedInstanceId: instanceId,
        expectedApiVersion: binary.identity.api_version,
        expectedNativeBuild: binary.identity.native_build,
        actualReady: health.ready,
        actualInstanceId: health.instanceId,
        actualApiVersion: health.apiVersion,
        actualNativeBuild: health.nativeBuild,
        output: currentOutput,
      });
    }
    yield* lifecycleLock.withPermits(1)(
      Effect.gen(function* () {
        const current = yield* SubscriptionRef.get(lifecycle)
        if (current._tag === "Exited") {
          const currentOutput = yield* Ref.get(output);
          return yield* new IcnExitedBeforeReady({
            pid: Number(process.pid),
            code: current.code,
            output: currentOutput,
          });
        }
        if (current._tag !== "Starting") {
          const currentOutput = yield* Ref.get(output);
          return yield* new IcnReadinessCommitRejected({
            output: currentOutput,
          });
        }
        yield* SubscriptionRef.set(
          lifecycle,
          IcnProcessLifecycleFsm.transition(current, "Ready", {}),
        )
      }),
    );

    const performShutdown = terminateProcess.pipe(
      Effect.zipRight(Deferred.await(exited)),
      Effect.asVoid,
    )
    const shutdown = Effect.uninterruptibleMask((restore) =>
      Effect.gen(function* () {
        const shouldStart = yield* lifecycleLock.withPermits(1)(
          Effect.gen(function* () {
            const current = yield* SubscriptionRef.get(lifecycle)
            if (current._tag === "Exited") {
              yield* Deferred.succeed(shutdownCompletion, undefined)
              return false
            }
            if (current._tag === "Stopping") return false
            yield* SubscriptionRef.set(
              lifecycle,
              IcnProcessLifecycleFsm.transition(current, "Stopping", {}),
            )
            return true
          }),
        );
        if (shouldStart) {
          yield* performShutdown.pipe(
            Effect.exit,
            Effect.flatMap((result) => Deferred.done(shutdownCompletion, result)),
            Effect.forkDaemon,
          );
        }
        return yield* restore(Deferred.await(shutdownCompletion));
      }),
    );
    const exit = Deferred.await(exited);
    return {
      process: IcnProcess.of({
        pid: Number(process.pid),
        origin,
        clientOptions: {
          baseUrl: origin,
          headers: { authorization: `Bearer ${authorization}` },
        },
        instanceId,
        binary,
        startup,
        diagnosticTail: Ref.get(output),
        lifecycle: SubscriptionRef.get(lifecycle),
        lifecycleChanges: lifecycle.changes,
        exit,
        unexpectedExit: exit.pipe(
          Effect.flatMap(({ code, diagnostic }) =>
            SubscriptionRef.get(lifecycle).pipe(
              Effect.flatMap((state) =>
                (state._tag === "Stopping" ||
                  (state._tag === "Exited" && state.expected))
                  ? Effect.never
                  : Effect.fail(new IcnUnexpectedExit({
                      pid: Number(process.pid),
                      code,
                      output: diagnostic,
                    }))
              )
            )
          )
        ),
        shutdown,
      }),
    };
  });

export const makeIcnProcess = (
  config: IcnLifecycleConfig
): Layer.Layer<
  IcnProcess,
  IcnLifecycleError,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | Path.Path
  | IcnPreparationReporterService
> =>
  Layer.scoped(
    IcnProcess,
    acquireIcn(config).pipe(
      Effect.map(({ process }) => process)
    )
  ).pipe(Layer.provideMerge(makeIcnBinaryResolver()));
