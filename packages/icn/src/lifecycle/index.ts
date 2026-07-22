import * as Command from "@effect/platform/Command";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import * as FileSystem from "@effect/platform/FileSystem";
import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientError from "@effect/platform/HttpClientError";
import * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import * as Path from "@effect/platform/Path";
import { GeneratedClientTransportError } from "@magnitudedev/openapi-effect/client-runtime";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import {
  Context,
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
  IcnApiClient,
  makeIcnApiClient,
  type IcnApiClient as IcnApiClientService,
} from "../generated/client.js";

const PositiveInt = Schema.Int.pipe(Schema.greaterThan(0));
const NonEmpty = Schema.String.pipe(Schema.minLength(1));

export const IcnBinarySource = Schema.Union(
  Schema.TaggedStruct("Explicit", { path: NonEmpty }),
  Schema.TaggedStruct("Release", {
    version: NonEmpty,
    platformKey: NonEmpty,
    dataDir: NonEmpty,
    releaseBaseUrl: NonEmpty,
  }),
  Schema.TaggedStruct("DevelopmentSearch", {
    candidates: Schema.NonEmptyArray(NonEmpty),
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
  allowBuildMismatch: Schema.Boolean,
  probeTimeout: Schema.DurationFromSelf.pipe(
    Schema.greaterThanDuration(Duration.zero)
  ),
  downloadTimeout: Schema.DurationFromSelf.pipe(
    Schema.greaterThanDuration(Duration.zero)
  ),
}) {}

export class IcnStorageConfig extends Schema.Class<IcnStorageConfig>(
  "IcnStorageConfig"
)({
  modelStore: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
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
});
export type IcnBinaryIdentity = typeof IcnBinaryIdentity.Type;

export interface ResolvedIcnBinary {
  readonly path: string;
  readonly identity: IcnBinaryIdentity;
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
  readonly cause?: unknown;
}> {}

const lifecycleError = (
  operation: IcnLifecycleOperation,
  reason: IcnLifecycleFailureReason,
  message: string,
  cause?: unknown
) =>
  new IcnLifecycleError({
    operation,
    reason,
    message,
    diagnostic: Option.none(),
    cause,
  });

const ReleaseManifest = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  binary: NonEmpty,
  sha256: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/)),
  apiVersion: PositiveInt,
  nativeBuild: NonEmpty,
  target: NonEmpty,
});

const releaseTag = (version: string) => `@magnitudedev/cli@${version}`;

export const icnReleaseAssetName = (platformKey: string) =>
  `magnitude-icn-${platformKey}.tar.gz`;

export const icnReleaseDownloadUrl = (
  releaseBaseUrl: string,
  version: string,
  platformKey: string
) =>
  `${releaseBaseUrl.replace(/\/+$/, "")}/${encodeURIComponent(
    releaseTag(version)
  )}/${icnReleaseAssetName(platformKey)}`;

const icnExecutableName = (platformKey: string) =>
  platformKey.startsWith("windows-") ? "magnitude-icn.exe" : "magnitude-icn";

const decodeReleaseManifest = (sourceText: string) =>
  Schema.decodeUnknown(Schema.parseJson(ReleaseManifest))(sourceText).pipe(
    Effect.mapError((cause) =>
      lifecycleError(
        "resolve",
        "invalid-manifest",
        "invalid ICN release manifest",
        cause
      )
    )
  );

const resolveCandidate = (
  source: IcnBinarySource,
  fs: FileSystem.FileSystem,
  path: Path.Path,
  executor: CommandExecutor.CommandExecutor,
  http: HttpClient.HttpClient,
  downloadTimeout: Duration.Duration
) =>
  Effect.gen(function* () {
    if (source._tag === "Explicit")
      return {
        path: source.path,
        manifest: Option.none<typeof ReleaseManifest.Type>(),
        install: Option.none<{
          staging: string;
          destination: string;
          version: string;
        }>(),
      };
    if (source._tag === "DevelopmentSearch") {
      for (const candidate of source.candidates) {
        if (yield* fs.exists(candidate).pipe(Effect.orElseSucceed(() => false)))
          return {
            path: candidate,
            manifest: Option.none<typeof ReleaseManifest.Type>(),
            install: Option.none<{
              staging: string;
              destination: string;
              version: string;
            }>(),
          };
      }
      return yield* lifecycleError(
        "resolve",
        "not-found",
        `none of the configured ICN development candidates exist`
      );
    }
    const executable = icnExecutableName(source.platformKey);
    const destination = path.join(
      source.dataDir,
      "bin",
      "icn",
      `${source.version}-${source.platformKey}`
    );
    const cachedBinary = path.join(destination, executable);
    const cachedManifest = path.join(
      destination,
      "magnitude-icn-manifest.json"
    );
    const cachedVersion = path.join(destination, "magnitude-icn.version");
    const cached = yield* Effect.all([
      fs.exists(cachedBinary).pipe(Effect.orElseSucceed(() => false)),
      fs.exists(cachedManifest).pipe(Effect.orElseSucceed(() => false)),
      fs.exists(cachedVersion).pipe(Effect.orElseSucceed(() => false)),
    ]).pipe(
      Effect.map(([binary, manifest, version]) => binary && manifest && version)
    );
    if (cached) {
      const version = yield* fs
        .readFileString(cachedVersion)
        .pipe(Effect.orElseSucceed(() => ""));
      const manifest = yield* fs.readFileString(cachedManifest).pipe(
        Effect.mapError((cause) =>
          lifecycleError(
            "resolve",
            "invalid-manifest",
            "unable to read the cached ICN release manifest",
            cause
          )
        ),
        Effect.flatMap(decodeReleaseManifest)
      );
      if (version.trim() === source.version && manifest.binary === executable)
        return {
          path: cachedBinary,
          manifest: Option.some(manifest),
          install: Option.none<{
            staging: string;
            destination: string;
            version: string;
          }>(),
        };
    }

    const nonce = (yield* Random.nextIntBetween(0, 0x1_0000_0000)).toString(16);
    const downloads = path.join(source.dataDir, "downloads");
    const staging = path.join(source.dataDir, "bin", `.icn-${nonce}`);
    const archive = path.join(
      downloads,
      `${icnReleaseAssetName(source.platformKey)}.${nonce}.tmp`
    );
    const url = icnReleaseDownloadUrl(
      source.releaseBaseUrl,
      source.version,
      source.platformKey
    );
    yield* fs
      .makeDirectory(downloads, { recursive: true })
      .pipe(
        Effect.mapError((cause) =>
          lifecycleError(
            "resolve",
            "download-failed",
            "unable to create the ICN download directory",
            cause
          )
        )
      );
    yield* fs
      .makeDirectory(staging, { recursive: true })
      .pipe(
        Effect.mapError((cause) =>
          lifecycleError(
            "resolve",
            "download-failed",
            "unable to create the ICN staging directory",
            cause
          )
        )
      );
    const response = yield* http.execute(HttpClientRequest.get(url)).pipe(
      Effect.retry({
        schedule: Schedule.exponential("1 second").pipe(
          Schedule.intersect(Schedule.recurs(2))
        ),
      }),
      Effect.timeoutFail({
        duration: downloadTimeout,
        onTimeout: () =>
          lifecycleError(
            "resolve",
            "download-failed",
            "ICN release download timed out"
          ),
      }),
      Effect.mapError((cause) =>
        cause instanceof IcnLifecycleError
          ? cause
          : lifecycleError(
              "resolve",
              "download-failed",
              `unable to download ${url}`,
              cause
            )
      )
    );
    if (response.status < 200 || response.status >= 300)
      return yield* lifecycleError(
        "resolve",
        "download-failed",
        `ICN release download returned HTTP ${response.status}`
      );
    const bytes = yield* response.arrayBuffer.pipe(
      Effect.mapError((cause) =>
        lifecycleError(
          "resolve",
          "download-failed",
          "unable to read the ICN release archive",
          cause
        )
      )
    );
    yield* fs
      .writeFile(archive, new Uint8Array(bytes))
      .pipe(
        Effect.mapError((cause) =>
          lifecycleError(
            "resolve",
            "download-failed",
            "unable to stage the ICN release archive",
            cause
          )
        )
      );
    const tarFlag = source.platformKey.startsWith("windows-") ? "-xf" : "-xzf";
    const extracted = yield* Command.exitCode(
      Command.make("tar", tarFlag, archive, "-C", staging)
    ).pipe(
      Effect.provideService(CommandExecutor.CommandExecutor, executor),
      Effect.mapError((cause) =>
        lifecycleError(
          "resolve",
          "invalid-archive",
          "unable to extract the ICN release archive",
          cause
        )
      ),
      Effect.ensuring(
        fs
          .remove(archive, { force: true })
          .pipe(Effect.catchAll(() => Effect.void))
      )
    );
    if (extracted !== 0)
      return yield* lifecycleError(
        "resolve",
        "invalid-archive",
        `ICN release archive extraction failed with ${extracted}`
      );
    const manifestPath = path.join(staging, "magnitude-icn-manifest.json");
    const manifest = yield* fs.readFileString(manifestPath).pipe(
      Effect.mapError((cause) =>
        lifecycleError(
          "resolve",
          "invalid-manifest",
          "ICN release archive has no readable manifest",
          cause
        )
      ),
      Effect.flatMap(decodeReleaseManifest)
    );
    if (manifest.binary !== executable)
      return yield* lifecycleError(
        "verify",
        "incompatible-build",
        "ICN release manifest does not match the requested platform"
      );
    return {
      path: path.join(staging, manifest.binary),
      manifest: Option.some(manifest),
      install: Option.some({ staging, destination, version: source.version }),
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
    const path = yield* Path.Path;
    const executor = yield* CommandExecutor.CommandExecutor;
    const http = yield* HttpClient.HttpClient;
    return IcnBinaryResolver.of({
      resolve: (config) =>
        Effect.suspend(() => {
          let stagingToClean: string | undefined;
          return Effect.gen(function* () {
            const candidate = yield* resolveCandidate(
              config.source,
              fs,
              path,
              executor,
              http,
              config.downloadTimeout
            );
            if (Option.isSome(candidate.install))
              stagingToClean = candidate.install.value.staging;
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
            if (Option.isSome(candidate.manifest)) {
              const bytes = yield* fs
                .readFile(canonical)
                .pipe(
                  Effect.mapError((cause) =>
                    lifecycleError(
                      "verify",
                      "checksum-mismatch",
                      "unable to hash the ICN binary",
                      cause
                    )
                  )
                );
              if (bytesToHex(sha256(bytes)) !== candidate.manifest.value.sha256)
                return yield* lifecycleError(
                  "verify",
                  "checksum-mismatch",
                  "ICN binary checksum does not match its release manifest"
                );
            }
            const output = yield* Command.string(
              Command.make(canonical, "version", "--json")
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
              Option.isSome(candidate.manifest) &&
              (identity.api_version !== candidate.manifest.value.apiVersion ||
                identity.native_build !==
                  candidate.manifest.value.nativeBuild ||
                identity.target !== candidate.manifest.value.target)
            )
              return yield* lifecycleError(
                "verify",
                "incompatible-build",
                "ICN binary identity does not match its companion manifest"
              );
            if (
              Option.isSome(config.expectedNativeBuild) &&
              identity.native_build !== config.expectedNativeBuild.value &&
              !config.allowBuildMismatch
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
            if (missing !== undefined)
              return yield* lifecycleError(
                "verify",
                "missing-capability",
                `ICN binary does not provide required capability ${missing}`
              );
            let published = canonical;
            if (Option.isSome(candidate.install)) {
              const { staging, destination, version } = candidate.install.value;
              yield* fs
                .writeFileString(
                  path.join(staging, "magnitude-icn.version"),
                  version
                )
                .pipe(
                  Effect.mapError((cause) =>
                    lifecycleError(
                      "resolve",
                      "download-failed",
                      "unable to write the ICN release cache marker",
                      cause
                    )
                  )
                );
              const destinationExists = yield* fs
                .exists(destination)
                .pipe(Effect.orElseSucceed(() => false));
              if (destinationExists) {
                yield* fs
                  .remove(staging, { recursive: true, force: true })
                  .pipe(
                    Effect.mapError((cause) =>
                      lifecycleError(
                        "resolve",
                        "download-failed",
                        "unable to discard a redundant ICN staging directory",
                        cause
                      )
                    )
                  );
              } else {
                yield* fs
                  .makeDirectory(path.dirname(destination), {
                    recursive: true,
                  })
                  .pipe(
                    Effect.mapError((cause) =>
                      lifecycleError(
                        "resolve",
                        "download-failed",
                        "unable to create the ICN cache directory",
                        cause
                      )
                    )
                  );
                yield* fs.rename(staging, destination).pipe(
                  Effect.catchAll((cause) =>
                    fs.exists(destination).pipe(
                      Effect.orElseSucceed(() => false),
                      Effect.flatMap((wonByPeer) =>
                        wonByPeer
                          ? fs
                              .remove(staging, {
                                recursive: true,
                                force: true,
                              })
                              .pipe(
                                Effect.mapError((removeCause) =>
                                  lifecycleError(
                                    "resolve",
                                    "download-failed",
                                    "unable to discard a redundant ICN staging directory",
                                    removeCause
                                  )
                                )
                              )
                          : Effect.fail(
                              lifecycleError(
                                "resolve",
                                "download-failed",
                                "unable to publish the verified ICN binary",
                                cause
                              )
                            )
                      )
                    )
                  )
                );
              }
              stagingToClean = undefined;
              published = yield* fs
                .realPath(
                  path.join(
                    destination,
                    candidate.manifest.pipe(Option.getOrThrow).binary
                  )
                )
                .pipe(
                  Effect.mapError((cause) =>
                    lifecycleError(
                      "resolve",
                      "not-found",
                      "unable to resolve the published ICN binary",
                      cause
                    )
                  )
                );
            }
            return { path: published, identity };
          }).pipe(
            Effect.onError(() =>
              stagingToClean === undefined
                ? Effect.void
                : fs
                    .remove(stagingToClean, { recursive: true, force: true })
                    .pipe(Effect.catchAll(() => Effect.void))
            )
          );
        }),
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

export interface IcnLifecycleService {
  readonly pid: number;
  readonly origin: URL;
  readonly instanceId: string;
  readonly binary: ResolvedIcnBinary;
  readonly startup: IcnStartupRecord;
  readonly diagnosticTail: Effect.Effect<string>;
  readonly exit: Effect.Effect<IcnExit, IcnLifecycleError>;
  readonly unexpectedExit: Effect.Effect<never, IcnLifecycleError>;
  readonly shutdownResult: Effect.Effect<void, IcnLifecycleError>;
}

export class IcnLifecycle extends Context.Tag("@magnitudedev/icn/IcnLifecycle")<
  IcnLifecycle,
  IcnLifecycleService
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
          diagnostic:
            diagnostic.trim() === "" ? Option.none() : Option.some(diagnostic),
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
  parentPid: number
): ReadonlyArray<string> => [
  "serve",
  "--bind",
  `${config.host === "::1" ? "[::1]" : config.host}:0`,
  "--instance-id",
  instanceId,
  "--parent-pid",
  String(parentPid),
  ...Option.match(config.storage.modelStore, {
    onNone: () => [],
    onSome: (value) => ["--model-store", value],
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
            ...renderIcnArguments(config, instanceId, config.parentPid)
          ).pipe(Command.env({ MAGNITUDE_ICN_AUTH_TOKEN: authorization }))
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
            Effect.catchAll(() => Effect.void)
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
      Effect.catchAll(() => Effect.void),
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
    const health = yield* client.system.health({}).pipe(
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

    const shutdown = Effect.gen(function* () {
      if (yield* Ref.getAndSet(stopping, true)) return;
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
    yield* Effect.addFinalizer(() =>
      shutdown.pipe(
        Effect.tap(() => Deferred.succeed(shutdownResult, undefined)),
        Effect.catchAll((error) => Deferred.fail(shutdownResult, error)),
        Effect.asVoid
      )
    );
    const exit = Deferred.await(exited);
    return {
      client,
      lifecycle: IcnLifecycle.of({
        pid: Number(process.pid),
        origin,
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
        shutdownResult: Deferred.await(shutdownResult),
      }),
      health,
    };
  });

export const makeIcn = (
  config: IcnLifecycleConfig
): Layer.Layer<
  IcnApiClient | IcnLifecycle,
  IcnLifecycleError,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | Path.Path
> =>
  Layer.scopedContext(
    acquireIcn(config).pipe(
      Effect.map(({ client, lifecycle }) =>
        Context.make(IcnApiClient, client as IcnApiClientService).pipe(
          Context.add(IcnLifecycle, lifecycle)
        )
      )
    )
  ).pipe(Layer.provideMerge(makeIcnBinaryResolver()));
