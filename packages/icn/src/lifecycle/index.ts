import * as Command from "@effect/platform/Command";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import * as FileSystem from "@effect/platform/FileSystem";
import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientError from "@effect/platform/HttpClientError";
import * as Path from "@effect/platform/Path";
import { GeneratedClientTransportError } from "@magnitudedev/openapi-effect/client-runtime";
import { delimiter, dirname, join } from "node:path";
import {
  Context,
  Cause,
  Data,
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
} from "effect";
import {
  makeIcnApiClient,
} from "../generated/client.js";
import { resolveReleaseIcnInstallation } from "./release-installation.js";

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
  modelSources: Schema.Array(NonEmpty),
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
  parentPid: PositiveInt,
}) {}

export const IcnBinaryIdentity = Schema.Struct({
  version: NonEmpty,
  api_version: PositiveInt,
  native_build: NonEmpty,
  target: NonEmpty,
  capabilities: Schema.Array(NonEmpty),
  backends: Schema.Array(NonEmpty),
});
export type IcnBinaryIdentity = typeof IcnBinaryIdentity.Type;

export interface ResolvedIcnBinary {
  readonly path: string;
  readonly identity: IcnBinaryIdentity;
  readonly installation: string;
  readonly environment: Readonly<Record<string, string>>;
}

export const IcnLifecycleOperation = Schema.Literal(
  "resolve",
  "verify",
  "spawn",
  "startup-record",
  "readiness",
  "observe-exit",
  "shutdown"
);
export type IcnLifecycleOperation = typeof IcnLifecycleOperation.Type;

export const IcnLifecycleFailureReason = Schema.Literal(
  "not-found",
  "invalid-configuration",
  "not-executable",
  "invalid-manifest",
  "probe-failed",
  "probe-timeout",
  "invalid-identity",
  "incompatible-api",
  "incompatible-build",
  "target-mismatch",
  "missing-capability",
  "checksum-mismatch",
  "download-failed",
  "invalid-archive",
  "spawn-failed",
  "invalid-startup-record",
  "startup-timeout",
  "exited-before-ready",
  "readiness-failed",
  "identity-mismatch",
  "unexpected-exit",
  "shutdown-failed"
);
export type IcnLifecycleFailureReason = typeof IcnLifecycleFailureReason.Type;

export class IcnLifecycleError extends Data.TaggedError("IcnLifecycleError")<{
  readonly operation: IcnLifecycleOperation;
  readonly reason: IcnLifecycleFailureReason;
  readonly message: string;
  readonly diagnostic: Option.Option<string>;
}> {}

const lifecycleError = <CauseValue>(
  operation: IcnLifecycleOperation,
  reason: IcnLifecycleFailureReason,
  message: string,
  ...cause: readonly [] | readonly [CauseValue]
) =>
  new IcnLifecycleError({
    operation,
    reason,
    message,
    diagnostic: Option.fromIterable(cause).pipe(
      Option.map((value) => Cause.pretty(Cause.fail(value))),
    ),
  });

const resolveCandidate = (
  source: IcnBinarySource,
) =>
  Effect.gen(function* () {
    if (source._tag === "Installation") {
      const root = dirname(source.path);
      const key = process.platform === "win32"
        ? "PATH"
        : process.platform === "darwin"
          ? "DYLD_LIBRARY_PATH"
          : "LD_LIBRARY_PATH";
      return {
        path: join(
          root,
          "bin",
          `magnitude-icn${process.platform === "win32" ? ".exe" : ""}`,
        ),
        installation: source.path,
        environment: {
          [key]: process.env[key]
            ? `${join(root, "runtime")}${delimiter}${process.env[key]}`
            : join(root, "runtime"),
        },
      };
    }
    const installation = yield* resolveReleaseIcnInstallation(
      source.version,
      source.dataDir,
      source.releaseBaseUrl,
    ).pipe(
      Effect.mapError((cause) =>
        lifecycleError(
          "resolve",
          "download-failed",
          "unable to prepare the release ICN installation",
          cause
        )
      )
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
  ) => Effect.Effect<ResolvedIcnBinary, IcnLifecycleError>;
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
    return IcnBinaryResolver.of({
      resolve: (config) =>
        Effect.suspend(() =>
          Effect.gen(function* () {
            const candidate = yield* resolveCandidate(config.source);
            const exists = yield* fs
              .exists(candidate.path)
              .pipe(Effect.orElseSucceed(() => false));
            if (!exists)
              return yield* lifecycleError(
                "resolve",
                "not-found",
                `ICN binary was not found at ${candidate.path}`
              );
            const canonical = yield* fs
              .realPath(candidate.path)
              .pipe(
                Effect.mapError((cause) =>
                  lifecycleError(
                    "resolve",
                    "not-found",
                    `unable to resolve ${candidate.path}`,
                    cause
                  )
                )
              );
            const info = yield* fs
              .stat(canonical)
              .pipe(
                Effect.mapError((cause) =>
                  lifecycleError(
                    "resolve",
                    "not-executable",
                    "unable to inspect the ICN binary",
                    cause
                  )
                )
              );
            if (
              info.type !== "File" ||
              (!canonical.toLowerCase().endsWith(".exe") &&
                (info.mode & 0o111) === 0)
            )
              return yield* lifecycleError(
                "resolve",
                "not-executable",
                "the resolved ICN binary is not executable"
              );
            const output = yield* Command.string(
              Command.make(canonical, "version", "--json").pipe(
                Command.env(candidate.environment)
              )
            ).pipe(
              Effect.provideService(CommandExecutor.CommandExecutor, executor),
              Effect.timeoutFail({
                duration: config.probeTimeout,
                onTimeout: () =>
                  lifecycleError(
                    "verify",
                    "probe-timeout",
                    "ICN identity probe timed out"
                  ),
              }),
              Effect.mapError((cause) =>
                cause instanceof IcnLifecycleError
                  ? cause
                  : lifecycleError(
                      "verify",
                      "probe-failed",
                      "ICN identity probe failed",
                      cause
                    )
              )
            );
            const identity = yield* Schema.decodeUnknown(
              Schema.parseJson(IcnBinaryIdentity)
            )(output).pipe(
              Effect.mapError((cause) =>
                cause instanceof IcnLifecycleError
                  ? cause
                  : lifecycleError(
                      "verify",
                      "invalid-identity",
                      "ICN identity did not match the protocol",
                      cause
                    )
              )
            );
            if (identity.api_version !== config.supportedApiVersion)
              return yield* lifecycleError(
                "verify",
                "incompatible-api",
                `ICN API ${identity.api_version} is incompatible with ${config.supportedApiVersion}`
              );
            if (
              Option.isSome(config.expectedNativeBuild) &&
              identity.native_build !== config.expectedNativeBuild.value
            )
              return yield* lifecycleError(
                "verify",
                "incompatible-build",
                "ICN native build does not match the release"
              );
            if (
              Option.isSome(config.expectedTarget) &&
              identity.target !== config.expectedTarget.value
            )
              return yield* lifecycleError(
                "verify",
                "target-mismatch",
                `ICN target ${identity.target} does not match ${config.expectedTarget.value}`
              );
            const missing = config.requiredCapabilities.find(
              (capability) => !identity.capabilities.includes(capability)
            );
            const missingCapability = Option.fromNullable(missing);
            if (Option.isSome(missingCapability))
              return yield* lifecycleError(
                "verify",
                "missing-capability",
                `ICN binary does not provide required capability ${missingCapability.value}`
              );
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

export const IcnStartupRecord = Schema.Struct({
  type: Schema.Literal("icn_ready"),
  protocolVersion: Schema.Literal(1),
  origin: NonEmpty,
  instanceId: NonEmpty,
  pid: PositiveInt,
  apiVersion: PositiveInt,
  nativeBuild: NonEmpty,
});
export type IcnStartupRecord = typeof IcnStartupRecord.Type;

export interface IcnExit {
  readonly code: number;
  readonly diagnostic: string;
}

export interface IcnProcessService {
  readonly pid: number;
  readonly origin: URL;
  readonly clientOptions: Parameters<typeof makeIcnApiClient>[0];
  readonly instanceId: string;
  readonly binary: ResolvedIcnBinary;
  readonly startup: IcnStartupRecord;
  readonly diagnosticTail: Effect.Effect<string>;
  readonly exit: Effect.Effect<IcnExit, IcnLifecycleError>;
  readonly unexpectedExit: Effect.Effect<never, IcnLifecycleError>;
  readonly shutdown: Effect.Effect<void, IcnLifecycleError>;
  readonly shutdownResult: Effect.Effect<void, IcnLifecycleError>;
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

const withDiagnostic = (error: IcnLifecycleError, output: Ref.Ref<string>) =>
  Ref.get(output).pipe(
    Effect.flatMap((diagnostic) =>
      Effect.fail(
        new IcnLifecycleError({
          ...error,
          diagnostic: diagnostic.trim() === ""
            ? error.diagnostic
            : Option.some(Option.match(error.diagnostic, {
                onNone: () => diagnostic,
                onSome: (cause) => `${cause}\n${diagnostic}`,
              })),
        })
      )
    )
  );

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
  parentPid: number,
  path: string,
): ReadonlyArray<string> => [
  "serve",
  "--bind",
  `${config.host === "::1" ? "[::1]" : config.host}:0`,
  "--instance-id",
  instanceId,
  "--parent-pid",
  String(parentPid),
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
  ...config.storage.modelSources.flatMap((value) => ["--model-source", value]),
  ...config.storage.huggingFaceCaches.flatMap((value) => ["--hf-cache", value]),
];

const acquireIcn = (input: IcnLifecycleConfig) =>
  Effect.gen(function* () {
    const config = yield* Schema.validate(IcnLifecycleConfig)(input).pipe(
      Effect.mapError((cause) =>
        lifecycleError(
          "resolve",
          "invalid-configuration",
          "invalid ICN lifecycle configuration",
          cause
        )
      )
    );
    const resolver = yield* IcnBinaryResolver;
    const binary = yield* resolver.resolve(config.binary);
    const instanceId = yield* opaqueInstanceId;
    const authorization = yield* opaqueInstanceId;
    const process = yield* Effect.uninterruptibleMask(() =>
      Effect.gen(function* () {
        const process = yield* Command.start(
          Command.make(
            binary.path,
            ...renderIcnArguments(
              config,
              instanceId,
              config.parentPid,
              binary.installation,
            )
          ).pipe(Command.env({
            ...binary.environment,
            MAGNITUDE_ICN_AUTH_TOKEN: authorization,
            HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
          }))
        ).pipe(
          Effect.mapError((cause) =>
            lifecycleError(
              "spawn",
              "spawn-failed",
              "failed to spawn ICN",
              cause
            )
          )
        );
        // Registration is atomic with spawn so interruption cannot orphan the child.
        yield* Effect.addFinalizer(() =>
          process.isRunning.pipe(
            Effect.flatMap((running) =>
              running ? process.kill("SIGTERM") : Effect.void
            ),
            Effect.option,
            Effect.asVoid,
          )
        );
        return process;
      })
    );
    const output = yield* Ref.make("");
    const startupRecord = yield* Deferred.make<
      IcnStartupRecord,
      IcnLifecycleError
    >();
    const exited = yield* Deferred.make<IcnExit, IcnLifecycleError>();
    const stopping = yield* Ref.make(false);
    const shutdownResult = yield* Deferred.make<void, IcnLifecycleError>();

    yield* process.stdout.pipe(
      Stream.decodeText(),
      Stream.splitLines,
      Stream.runForEach((line) =>
        Effect.gen(function* () {
          yield* appendBounded(output, `${line}\n`, config.outputLimitBytes);
          if (!line.startsWith("MAGNITUDE_ICN_READY ")) return;
          const record = yield* Schema.decodeUnknown(
            Schema.parseJson(IcnStartupRecord)
          )(line.slice("MAGNITUDE_ICN_READY ".length)).pipe(
            Effect.mapError((cause) =>
              cause instanceof IcnLifecycleError
                ? cause
                : lifecycleError(
                    "startup-record",
                    "invalid-startup-record",
                    "invalid startup record",
                    cause
                  )
            )
          );
          yield* Deferred.complete(startupRecord, Effect.succeed(record));
        }).pipe(Effect.catchAll((error) => Deferred.fail(startupRecord, error)))
      ),
      Effect.catchAll((cause) =>
        Deferred.fail(
          startupRecord,
          lifecycleError(
            "startup-record",
            "invalid-startup-record",
            "stdout closed before startup",
            cause
          )
        )
      ),
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
            Deferred.succeed(exited, { code, diagnostic })
          )
        )
      ),
      Effect.catchAll((cause) =>
        Deferred.fail(
          exited,
          lifecycleError(
            "observe-exit",
            "unexpected-exit",
            "failed to observe ICN exit",
            cause
          )
        )
      ),
      Effect.forkScoped
    );

    const earlyExit = Deferred.await(exited).pipe(
      Effect.flatMap(({ code }) =>
        Effect.fail(
          lifecycleError(
            "startup-record",
            "exited-before-ready",
            `ICN exited with ${code} before readiness`
          )
        )
      )
    );
    const startup = yield* Effect.raceFirst(
      Deferred.await(startupRecord),
      earlyExit
    ).pipe(
      Effect.timeoutFail({
        duration: config.startupTimeout,
        onTimeout: () =>
          lifecycleError(
            "startup-record",
            "startup-timeout",
            "ICN startup record timed out"
          ),
      }),
      Effect.catchAll((error) => withDiagnostic(error, output))
    );
    if (
      startup.instanceId !== instanceId ||
      startup.pid !== Number(process.pid) ||
      startup.apiVersion !== binary.identity.api_version ||
      startup.nativeBuild !== binary.identity.native_build
    )
      return yield* withDiagnostic(
        lifecycleError(
          "startup-record",
          "identity-mismatch",
          "ICN startup identity does not match its owner or binary"
        ),
        output
      );
    const origin = yield* Effect.try({
      try: () => new URL(startup.origin),
      catch: (cause) =>
        lifecycleError(
          "startup-record",
          "invalid-startup-record",
          "ICN startup origin is invalid",
          cause
        ),
    });
    if (
      (origin.hostname !== "127.0.0.1" &&
        origin.hostname !== "[::1]" &&
        origin.hostname !== "::1") ||
      origin.protocol !== "http:"
    )
      return yield* lifecycleError(
        "startup-record",
        "invalid-startup-record",
        "ICN did not bind a loopback HTTP origin"
      );
    const client = yield* makeIcnApiClient({
      baseUrl: origin,
      headers: { authorization: `Bearer ${authorization}` },
    });
    yield* client.system.health({}).pipe(
      Effect.flatMap((value) =>
        value.ready &&
        value.instanceId === instanceId &&
        value.apiVersion === binary.identity.api_version &&
        value.nativeBuild === binary.identity.native_build
          ? Effect.succeed(value)
          : Effect.fail(
              lifecycleError(
                "readiness",
                "identity-mismatch",
                "ICN health identity does not match startup"
              )
            )
      ),
      Effect.retry({
        schedule: Schedule.spaced("50 millis"),
        while: (cause) =>
          cause instanceof GeneratedClientTransportError &&
          cause.cause instanceof HttpClientError.RequestError,
      }),
      Effect.mapError((cause) =>
        cause instanceof IcnLifecycleError
          ? cause
          : lifecycleError(
              "readiness",
              "readiness-failed",
              "ICN readiness probe failed",
              cause
            )
      ),
      Effect.timeoutFail({
        duration: config.startupTimeout,
        onTimeout: () =>
          lifecycleError(
            "readiness",
            "startup-timeout",
            "ICN readiness timed out"
          ),
      }),
      Effect.catchAll((error) => withDiagnostic(error, output))
    );

    const performShutdown = Effect.gen(function* () {
      const alreadyExited = yield* Deferred.isDone(exited);
      if (!alreadyExited) {
        yield* process
          .kill("SIGTERM")
          .pipe(
            Effect.mapError((cause) =>
              lifecycleError(
                "shutdown",
                "shutdown-failed",
                "failed to terminate ICN",
                cause
              )
            )
          );
        const graceful = yield* Deferred.await(exited).pipe(
          Effect.timeoutOption(config.gracefulShutdownTimeout)
        );
        if (Option.isNone(graceful)) {
          yield* process
            .kill("SIGKILL")
            .pipe(
              Effect.mapError((cause) =>
                lifecycleError(
                  "shutdown",
                  "shutdown-failed",
                  "failed to force-kill ICN",
                  cause
                )
              )
            );
          yield* Deferred.await(exited).pipe(
            Effect.timeoutFail({
              duration: config.forceShutdownTimeout,
              onTimeout: () =>
                lifecycleError(
                  "shutdown",
                  "shutdown-failed",
                  "ICN did not exit after force-kill"
                ),
            })
          );
        }
      }
    });
    const shutdown = Effect.gen(function* () {
      const alreadyStopping = yield* Ref.getAndSet(stopping, true);
      if (alreadyStopping) return yield* Deferred.await(shutdownResult);

      const result = yield* performShutdown.pipe(Effect.exit);
      yield* Deferred.done(shutdownResult, result);
      return yield* Deferred.await(shutdownResult);
    });
    yield* Effect.addFinalizer(() =>
      shutdown.pipe(Effect.ignore)
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
        exit,
        unexpectedExit: exit.pipe(
          Effect.flatMap(({ code }) =>
            Ref.get(stopping).pipe(
              Effect.flatMap((expected) =>
                expected
                  ? Effect.never
                  : Effect.fail(
                      lifecycleError(
                        "observe-exit",
                        "unexpected-exit",
                        `ICN exited unexpectedly with ${code}`
                      )
                    )
              )
            )
          )
        ),
        shutdown,
        shutdownResult: Deferred.await(shutdownResult),
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
> =>
  Layer.scoped(
    IcnProcess,
    acquireIcn(config).pipe(
      Effect.map(({ process }) => process)
    )
  ).pipe(Layer.provideMerge(makeIcnBinaryResolver()));
