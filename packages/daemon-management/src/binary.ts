import * as Command from "@effect/platform/Command";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import * as FileSystem from "@effect/platform/FileSystem";
import * as HttpClient from "@effect/platform/HttpClient";
import * as Path from "@effect/platform/Path";
import { Array as Arr, Effect, Option } from "effect";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  acquireRelease,
  currentHost,
  installArtifact,
  ACN_EXECUTABLE_NAME,
  NodeArchiveExtractor,
  releaseBaseUrl,
  releaseBundleSizes,
  releaseTag,
  selectArtifact,
  type ArtifactInstallationEvent,
  type ReleaseBundleSizes,
} from "@magnitudedev/release";
import {
  BinaryNotFound,
  BinaryRevisionMismatch,
  BinaryVersionMismatch,
  AcnEnsuranceFailed,
  DownloadFailed,
} from "./errors";

export type BinaryAcquisitionEvent =
  | {
      readonly _tag: "Planned";
      readonly plan: ReleaseBundleSizes;
    }
  | {
      readonly _tag: "Artifact";
      readonly event: ArtifactInstallationEvent;
    };

export interface ResolveBinaryOptions {
  readonly binaryPath?: string;
  readonly version?: string;
  readonly acnRevision?: number;
  readonly dataDir?: string;
  readonly acquisitionObserver: Option.Option<{
    readonly report: (event: BinaryAcquisitionEvent) => Effect.Effect<void>;
  }>;
}

export interface ResolvedBinaryCommand {
  readonly command: Arr.NonEmptyReadonlyArray<string>;
  readonly needsDownload: boolean;
}

export const defaultDataDir = (): string => join(homedir(), ".magnitude");
export const defaultBinaryPath = (dataDir: string = defaultDataDir()): string =>
  join(dataDir, "bin", ACN_EXECUTABLE_NAME);

const releaseRoot = (dataDir: string) => join(dataDir, "releases");
const acnRoot = (dataDir: string, version: string) =>
  join(releaseRoot(dataDir), "acn", version, currentHost());
const pointerPath = (dataDir: string, version: string) =>
  join(acnRoot(dataDir, version), "current.txt");
const executableName = () =>
  `${ACN_EXECUTABLE_NAME}${process.platform === "win32" ? ".exe" : ""}`;

const validateBinaryVersion = (
  binaryPath: string,
  expectedVersion: string
): Effect.Effect<
  void,
  BinaryVersionMismatch | AcnEnsuranceFailed,
  CommandExecutor.CommandExecutor
> =>
  Effect.gen(function* () {
    const actual = yield* Command.make(binaryPath, "version").pipe(
      Command.string,
      Effect.map((value) => value.trim()),
      Effect.mapError(
        (cause) => new AcnEnsuranceFailed({ reason: String(cause) })
      )
    );
    if (actual !== expectedVersion) {
      return yield* new BinaryVersionMismatch({
        path: binaryPath,
        expected: expectedVersion,
        actual,
      });
    }
  });

const validateBinaryRevision = (
  binaryPath: string,
  expectedRevision: number,
): Effect.Effect<
  void,
  BinaryRevisionMismatch | AcnEnsuranceFailed,
  CommandExecutor.CommandExecutor
> => Effect.gen(function* () {
  const output = yield* Command.make(binaryPath, "coordination-revision").pipe(
    Command.string,
    Effect.map((value) => value.trim()),
    Effect.mapError((cause) => new AcnEnsuranceFailed({ reason: String(cause) })),
  )
  const actual = Number(output)
  if (!Number.isSafeInteger(actual) || actual <= 0) {
    return yield* new AcnEnsuranceFailed({
      reason: `Magnitude executable ${binaryPath} returned an invalid ACN revision`,
    })
  }
  if (actual !== expectedRevision) {
    return yield* new BinaryRevisionMismatch({
      path: binaryPath,
      expected: expectedRevision,
      actual,
    })
  }
})

const cachedAcn = (
  dataDir: string,
  version: string,
  expectedRevision: number | undefined,
): Effect.Effect<
  Option.Option<string>,
  never,
  CommandExecutor.CommandExecutor | FileSystem.FileSystem | Path.Path
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const digest = yield* fs.readFileString(pointerPath(dataDir, version)).pipe(
      Effect.map((value) => value.trim()),
      Effect.orElseSucceed(() => "")
    );
    if (!/^[a-f0-9]{64}$/.test(digest)) return Option.none();
    const executable = path.join(
      acnRoot(dataDir, version),
      digest,
      "bin",
      executableName()
    );
    if (
      !(yield* fs.exists(executable).pipe(Effect.orElseSucceed(() => false)))
    ) {
      return Option.none();
    }
    const valid = yield* validateBinaryVersion(executable, version).pipe(
      Effect.zipRight(expectedRevision === undefined
        ? Effect.void
        : validateBinaryRevision(executable, expectedRevision)),
      Effect.as(true),
      Effect.catchAll(() => Effect.succeed(false))
    );
    return valid ? Option.some(executable) : Option.none();
  });

const publishPointer = (
  dataDir: string,
  version: string,
  digest: string
): Effect.Effect<void, DownloadFailed, FileSystem.FileSystem | Path.Path> =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const pointer = pointerPath(dataDir, version);
      const parent = path.dirname(pointer);
      yield* fs
        .makeDirectory(parent, { recursive: true, mode: 0o700 })
        .pipe(Effect.mapError(acquisitionFailure(version)));
      const temporary = yield* fs
        .makeTempFileScoped({
          directory: parent,
          prefix: ".current-",
        })
        .pipe(Effect.mapError(acquisitionFailure(version)));
      yield* fs
        .writeFileString(temporary, `${digest}\n`, { mode: 0o600 })
        .pipe(Effect.mapError(acquisitionFailure(version)));
      yield* fs.rename(temporary, pointer).pipe(
        Effect.catchAll(() =>
          fs
            .remove(pointer, { force: true })
            .pipe(Effect.zipRight(fs.rename(temporary, pointer)))
        ),
        Effect.mapError(acquisitionFailure(version))
      );
    })
  );

const acquisitionFailure = (version: string) => (cause: unknown) =>
  new DownloadFailed({
    url: `${releaseBaseUrl()}/${encodeURIComponent(releaseTag(version))}`,
    status: 0,
    reason: cause instanceof Error ? cause.message : String(cause),
  });

const ensureAcn = (
  version: string,
  expectedRevision: number | undefined,
  dataDir: string,
  observer: ResolveBinaryOptions["acquisitionObserver"]
): Effect.Effect<
  { readonly path: string; readonly acquired: boolean },
  DownloadFailed | BinaryVersionMismatch | BinaryRevisionMismatch | AcnEnsuranceFailed,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const cached = yield* cachedAcn(dataDir, version, expectedRevision);
    if (Option.isSome(cached)) return { path: cached.value, acquired: false };

    const manifestPath = path.join(releaseRoot(dataDir), "manifests", version)
    const release = yield* acquireRelease(
      releaseBaseUrl(),
      version,
      manifestPath,
    ).pipe(Effect.mapError(acquisitionFailure(version)));
    if (expectedRevision !== undefined && release.manifest.acnRevision !== expectedRevision) {
      return yield* new BinaryRevisionMismatch({
        path: manifestPath,
        expected: expectedRevision,
        actual: release.manifest.acnRevision,
      })
    }
    const artifact = yield* selectArtifact(
      release.manifest,
      "acn",
      currentHost()
    ).pipe(Effect.mapError(acquisitionFailure(version)));
    const plan = yield* releaseBundleSizes(
      release.manifest,
      currentHost()
    ).pipe(Effect.mapError(acquisitionFailure(version)));
    yield* Option.match(observer, {
      onNone: () => Effect.void,
      onSome: ({ report }) => report({ _tag: "Planned", plan }),
    });
    const destination = path.join(acnRoot(dataDir, version), artifact.sha256);
    const executable = path.join(destination, "bin", executableName());

    if (yield* fs.exists(destination).pipe(Effect.orElseSucceed(() => false))) {
      const valid = yield* validateBinaryVersion(executable, version).pipe(
        Effect.as(true),
        Effect.catchAll(() => Effect.succeed(false))
      );
      if (!valid) {
        yield* fs
          .remove(destination, { recursive: true, force: true })
          .pipe(Effect.mapError(acquisitionFailure(version)));
      }
    }
    let acquired = false;
    if (
      !(yield* fs.exists(destination).pipe(Effect.orElseSucceed(() => false)))
    ) {
      yield* installArtifact(releaseBaseUrl(), version, artifact, destination, {
        observer: Option.map(observer, ({ report }) => ({
          report: (event) => report({ _tag: "Artifact", event }),
        })),
      }).pipe(
        Effect.provide(NodeArchiveExtractor),
        Effect.mapError(acquisitionFailure(version))
      );
      acquired = true;
    }
    yield* validateBinaryVersion(executable, version);
    if (expectedRevision !== undefined) {
      yield* validateBinaryRevision(executable, expectedRevision)
    }
    yield* publishPointer(dataDir, version, artifact.sha256);
    return { path: executable, acquired };
  });

/**
 * Whether an installed ACN build for this version is present locally: the
 * published pointer names a digest whose executable exists. A pure file probe
 * with no subprocess validation, so the ensure pipeline remains the authority
 * — a corrupted build can answer true and still be reinstalled, a running
 * daemon whose files were removed can answer false and still be adopted.
 * Callers use it to schedule around a likely installation, not to decide one.
 */
export const acnInstallationPresent = (
  version: string,
  dataDir: string = defaultDataDir()
): Effect.Effect<boolean, never, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const digest = yield* fs.readFileString(pointerPath(dataDir, version)).pipe(
      Effect.map((value) => value.trim()),
      Effect.orElseSucceed(() => "")
    );
    if (!/^[a-f0-9]{64}$/.test(digest)) return false;
    const executable = path.join(
      acnRoot(dataDir, version),
      digest,
      "bin",
      executableName()
    );
    return yield* fs.exists(executable).pipe(Effect.orElseSucceed(() => false));
  });

export const downloadAcn = (
  version: string,
  dataDir: string
): Effect.Effect<
  string,
  DownloadFailed | BinaryVersionMismatch | BinaryRevisionMismatch | AcnEnsuranceFailed,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
> =>
  ensureAcn(version, undefined, dataDir, Option.none()).pipe(
    Effect.map(({ path }) => path)
  );

export const resolveBinaryCommand = (
  options: ResolveBinaryOptions = {
    acquisitionObserver: Option.none(),
  }
): Effect.Effect<
  ResolvedBinaryCommand,
  DownloadFailed | BinaryNotFound | BinaryVersionMismatch | BinaryRevisionMismatch | AcnEnsuranceFailed,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const dataDir = options?.dataDir ?? defaultDataDir();
    const expectedVersion = options?.version;
    const expectedRevision = options?.acnRevision;

    if (options?.binaryPath) {
      if (
        !(yield* fs
          .exists(options.binaryPath)
          .pipe(Effect.orElseSucceed(() => false)))
      ) {
        return yield* new BinaryNotFound({ path: options.binaryPath });
      }
      if (expectedVersion)
        yield* validateBinaryVersion(options.binaryPath, expectedVersion);
      if (expectedRevision !== undefined)
        yield* validateBinaryRevision(options.binaryPath, expectedRevision);
      return {
        command: [
          options.binaryPath,
          "serve",
        ],
        needsDownload: false,
      };
    }

    if (!expectedVersion) {
      const path = defaultBinaryPath(dataDir);
      if (!(yield* fs.exists(path).pipe(Effect.orElseSucceed(() => false)))) {
        return yield* new BinaryNotFound({ path });
      }
      return {
        command: [path, "serve"],
        needsDownload: false,
      };
    }

    const resolved = yield* ensureAcn(
      expectedVersion,
      expectedRevision,
      dataDir,
      options.acquisitionObserver
    );
    return {
      command: [resolved.path, "serve"],
      needsDownload: resolved.acquired,
    };
  });
