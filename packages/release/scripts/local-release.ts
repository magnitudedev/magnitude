import { resolve } from "node:path"
import releasePlan from "../release-plan.json"
import * as FileSystem from "@effect/platform/FileSystem"
import { Console, Data, Effect, Option, Schema } from "effect"
import {
  releaseTag,
  ReleaseArtifactSchema,
  ReleaseManifestSchema,
  type ReleaseArtifact,
  type ReleaseManifest,
} from "../src/contracts"
import { ACN_COORDINATION_REVISION } from "@magnitudedev/version"
import {
  acnArchive,
  backendPacks,
  cliArchive,
  currentHost,
  hostById,
  type ReleaseHost,
} from "../src/targets"
import { buildAcnBinary } from "./build/acn"
import { buildBackendArtifact } from "./build/backend"
import { buildCliBinary } from "./build/cli"
import { buildHostArtifacts } from "./build/host"
import { buildArchive, run } from "./build/common"
import { ACN_EXECUTABLE_NAME } from "../src/executables"

/**
 * A release built from the local worktree instead of published: byte-for-byte
 * the file set a GitHub release of this code would contain — the manifest plus
 * every artifact archive for the current host — existing only on this machine
 * so dev harnesses can serve it as a release origin.
 */
export interface LocalRelease {
  readonly version: string
  readonly files: ReadonlyMap<string, string>
}

export class LocalReleaseError extends Data.TaggedError("LocalReleaseError")<{
  readonly message: string
}> {}

const PROJECT_ROOT = resolve(import.meta.dir, "../../..")
const CATALOG_ROOT = resolve(PROJECT_ROOT, "inference/target/catalog-inputs")
const VERSION_FILE = resolve(
  PROJECT_ROOT,
  "packages/version/src/version.generated.ts",
)

// Directory name predates the LocalRelease terminology; renaming it would
// orphan cached ICN builds.
export const LOCAL_RELEASE_BUILD_ROOT = resolve(
  PROJECT_ROOT,
  "inference/target/release-bootstrap-test",
)

const localReleaseFailure = (message: string) => new LocalReleaseError({ message })

export const localReleaseBuildStep = <A>(message: string, evaluate: () => Promise<A>) =>
  Effect.tryPromise({
    try: evaluate,
    catch: (cause) => localReleaseFailure(`${message}: ${String(cause)}`),
  })

export const runRepoCommand = (
  executable: string,
  ...arguments_: readonly string[]
): Effect.Effect<string, LocalReleaseError> =>
  localReleaseBuildStep(
    `command failed: ${[executable, ...arguments_].join(" ")}`,
    () => run([executable, ...arguments_], { cwd: PROJECT_ROOT }),
  )

const readPackageVersion = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const packageJson = yield* fs.readFileString(
    resolve(PROJECT_ROOT, "packages/launcher/package.json"),
  ).pipe(
    Effect.mapError((cause) =>
      localReleaseFailure(`unable to read the CLI package: ${String(cause)}`)
    ),
  )
  const decoded = yield* Schema.decodeUnknown(
    Schema.parseJson(Schema.Struct({ version: Schema.NonEmptyString })),
  )(packageJson).pipe(
    Effect.mapError((cause) =>
      localReleaseFailure(`CLI package has an invalid version: ${String(cause)}`)
    ),
  )
  return decoded.version
})

const readArtifact = (
  descriptor: string,
): Effect.Effect<ReleaseArtifact, LocalReleaseError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const contents = yield* fs.readFileString(descriptor).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`unable to read ${descriptor}: ${String(cause)}`)
      ),
    )
    return yield* Schema.decodeUnknown(
      Schema.parseJson(ReleaseArtifactSchema),
    )(contents).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`invalid artifact descriptor ${descriptor}: ${String(cause)}`)
      ),
    )
  })

const readManifestFile = (
  path: string,
): Effect.Effect<ReleaseManifest, LocalReleaseError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const contents = yield* fs.readFileString(path).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`unable to read ${path}: ${String(cause)}`)
      ),
    )
    return yield* Schema.decodeUnknown(
      Schema.parseJson(ReleaseManifestSchema),
    )(contents).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`invalid release manifest ${path}: ${String(cause)}`)
      ),
    )
  })

const writeManifestFile = (
  path: string,
  manifest: ReleaseManifest,
): Effect.Effect<void, LocalReleaseError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const serialized = yield* Schema.encode(
      Schema.parseJson(ReleaseManifestSchema),
    )(manifest).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`unable to encode the release manifest: ${String(cause)}`)
      ),
    )
    yield* fs.writeFileString(path, `${serialized}\n`).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`unable to write ${path}: ${String(cause)}`)
      ),
    )
  })

/** Builds the CLI and ACN archives (the only version-stamped artifacts). */
const packageRuntimeArtifacts = (
  host: ReleaseHost,
  outputRoot: string,
  binaries: { readonly cli: string; readonly acn: string },
): Effect.Effect<
  { readonly cli: ReleaseArtifact; readonly acn: ReleaseArtifact },
  LocalReleaseError,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const cliArchivePath = resolve(outputRoot, cliArchive(host.id))
    const cliDescriptorPath = resolve(outputRoot, `cli-${host.id}.artifact.json`)
    const acnArchivePath = resolve(outputRoot, acnArchive(host.id))
    const acnDescriptorPath = resolve(outputRoot, `acn-${host.id}.artifact.json`)
    yield* Effect.all(
      [cliArchivePath, cliDescriptorPath, acnArchivePath, acnDescriptorPath]
        .map((path) =>
          fs.remove(path, { force: true }).pipe(
            Effect.mapError((cause) =>
              localReleaseFailure(`unable to replace ${path}: ${String(cause)}`)
            ),
          )
        ),
      { discard: true },
    )
    const cli = yield* localReleaseBuildStep(
      `unable to package the ${host.id} CLI`,
      () =>
        buildArchive(cliArchivePath, cliDescriptorPath, {
          id: `cli-${host.id}`,
          kind: "cli",
          host: Option.some(host.id),
          backend: Option.none(),
          requiredBaseId: Option.none(),
          nativeBuild: Option.none(),
          backendModuleAbi: Option.none(),
          compatibility: Option.none(),
        }, [{
          path: `bin/magnitude-cli${host.executableExtension}`,
          source: binaries.cli,
          mode: 0o755,
        }]),
    )
    const acn = yield* localReleaseBuildStep(
      `unable to package the ${host.id} ACN`,
      () =>
        buildArchive(acnArchivePath, acnDescriptorPath, {
          id: `acn-${host.id}`,
          kind: "acn",
          host: Option.some(host.id),
          backend: Option.none(),
          requiredBaseId: Option.none(),
          nativeBuild: Option.none(),
          backendModuleAbi: Option.none(),
          compatibility: Option.none(),
        }, [{
          path: `bin/${ACN_EXECUTABLE_NAME}${host.executableExtension}`,
          source: binaries.acn,
          mode: 0o755,
        }]),
    )
    return { cli, acn }
  })

const replaceRuntimeArtifacts = (
  manifest: ReleaseManifest,
  runtime: { readonly cli: ReleaseArtifact; readonly acn: ReleaseArtifact },
): Effect.Effect<ReleaseManifest["artifacts"], LocalReleaseError> => {
  const updated = manifest.artifacts.map((current) =>
    current.id === runtime.cli.id
      ? runtime.cli
      : current.id === runtime.acn.id
        ? runtime.acn
        : current
  )
  const [first, ...remaining] = updated
  return first
    ? Effect.succeed([first, ...remaining] as const)
    : Effect.fail(localReleaseFailure("local release has no artifacts"))
}

export const buildLocalRelease: Effect.Effect<
  LocalRelease,
  LocalReleaseError,
  FileSystem.FileSystem
> = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const host = currentHost()
  const root = resolve(LOCAL_RELEASE_BUILD_ROOT, host)
  const hostRoot = resolve(root, "host")
  const packs = backendPacks.filter(
    (pack) => pack.host === host && pack.backend === "metal",
  )

  yield* Console.log("Building model planner inputs")
  yield* runRepoCommand("bun", "run", "icn:catalog:build-bundle")

  yield* Console.log(`Building ${host} CLI, ACN, and ICN release artifacts`)
  yield* localReleaseBuildStep(
    `unable to build ${host} release artifacts`,
    () => buildHostArtifacts(host, CATALOG_ROOT, hostRoot),
  )

  for (const pack of packs) {
    const packRoot = resolve(root, pack.id)
    yield* Console.log(`Building ${pack.id} release artifact`)
    yield* localReleaseBuildStep(
      `unable to build ${pack.id}`,
      () => buildBackendArtifact(pack.id, packRoot),
    )
  }

  const artifactRoots = [
    hostRoot,
    ...packs.map((pack) => resolve(root, pack.id)),
  ]
  const descriptors = (
    yield* Effect.forEach(artifactRoots, (artifactRoot) =>
      fs.readDirectory(artifactRoot).pipe(
        Effect.map((entries) =>
          entries
            .filter((entry) => entry.endsWith(".artifact.json"))
            .map((entry) => resolve(artifactRoot, entry))
        ),
        Effect.mapError((cause) =>
          localReleaseFailure(`unable to inspect ${artifactRoot}: ${String(cause)}`)
        ),
      )
    )
  ).flat()
  const artifacts = yield* Effect.forEach(descriptors, readArtifact)
  const byId = new Map(artifacts.map((artifact) => [artifact.id, artifact]))
  const base = byId.get(`icn-base-${host}`)
  if (!base) return yield* localReleaseFailure(`local release has no ICN base for ${host}`)
  for (const pack of artifacts.filter(
    (artifact) => artifact.kind === "icn-backend",
  )) {
    if (
      Option.getOrUndefined(pack.requiredBaseId) !== base.id ||
      Option.getOrUndefined(pack.nativeBuild) !==
        Option.getOrUndefined(base.nativeBuild) ||
      Option.getOrUndefined(pack.backendModuleAbi) !==
        Option.getOrUndefined(base.backendModuleAbi)
    ) {
      return yield* localReleaseFailure(`${pack.id} is incompatible with ${base.id}`)
    }
  }

  const version = yield* readPackageVersion
  const sourceCommit = (yield* runRepoCommand("git", "rev-parse", "HEAD")).trim()
  const sortedArtifacts = artifacts
    .slice()
    .sort((left, right) => left.id.localeCompare(right.id))
  const [firstArtifact, ...remainingArtifacts] = sortedArtifacts
  if (!firstArtifact) {
    return yield* localReleaseFailure("local release has no artifacts")
  }
  const manifest = {
    schemaVersion: 2,
    version,
    acnRevision: ACN_COORDINATION_REVISION,
    rpc: releasePlan.rpc,
    plugins: yield* Schema.decodeUnknown(ReleaseManifestSchema.fields.plugins)(releasePlan.plugins.map(plugin => plugin.artifact)).pipe(Effect.mapError(error => localReleaseFailure(`Invalid plugin inventory: ${String(error)}`))),
    tag: releaseTag(version),
    sourceCommit,
    artifacts: [firstArtifact, ...remainingArtifacts],
  } satisfies ReleaseManifest
  const manifestPath = resolve(root, "magnitude-release.json")
  yield* writeManifestFile(manifestPath, manifest)

  const files = new Map<string, string>([
    ["magnitude-release.json", manifestPath],
  ])
  for (const artifact of artifacts) {
    const artifactRoot = artifactRoots.find((root) =>
      descriptors.includes(resolve(root, `${artifact.id}.artifact.json`))
    )
    if (!artifactRoot) {
      return yield* localReleaseFailure(`unable to locate ${artifact.id}`)
    }
    files.set(artifact.filename, resolve(artifactRoot, artifact.filename))
  }
  return { version, files } satisfies LocalRelease
})

export const loadLocalRelease: Effect.Effect<
  LocalRelease,
  LocalReleaseError,
  FileSystem.FileSystem
> = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const host = currentHost()
  const root = resolve(LOCAL_RELEASE_BUILD_ROOT, host)
  const version = yield* readPackageVersion
  const manifestPath = resolve(root, "magnitude-release.json")
  const manifest = yield* readManifestFile(manifestPath).pipe(
    Effect.mapError((error) =>
      localReleaseFailure(`no local release exists: ${error.message}`)
    ),
  )
  if (manifest.version !== version) {
    return yield* localReleaseFailure(
      `cached local release is ${manifest.version}, but the CLI is ${version}`,
    )
  }
  const files = new Map<string, string>([
    ["magnitude-release.json", manifestPath],
  ])
  for (const artifact of manifest.artifacts) {
    const locations = [
      resolve(root, "host", artifact.filename),
      ...backendPacks
        .filter((pack) => pack.host === host)
        .map((pack) => resolve(root, pack.id, artifact.filename)),
    ]
    const existing = yield* Effect.findFirst(locations, (location) =>
      fs.exists(location)
    ).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`unable to inspect local release files: ${String(cause)}`)
      ),
    )
    if (Option.isNone(existing)) {
      return yield* localReleaseFailure(
        `cached local release is missing ${artifact.filename}`,
      )
    }
    files.set(artifact.filename, existing.value)
  }
  return { version, files } satisfies LocalRelease
})

/** Rebuilds a local release's CLI and ACN binaries from the current worktree. */
export const refreshLocalRelease = (
  release: LocalRelease,
): Effect.Effect<LocalRelease, LocalReleaseError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const host = hostById(currentHost())
    const root = resolve(LOCAL_RELEASE_BUILD_ROOT, host.id)
    const hostRoot = resolve(root, "host")
    const manifestPath = resolve(root, "magnitude-release.json")
    const manifest = yield* readManifestFile(manifestPath)

    yield* runRepoCommand("bun", "run", "packages/version/scripts/generate-version.ts")
    yield* Console.log(`Refreshing the ${host.id} CLI and ACN release artifacts`)
    const cliBinary = yield* localReleaseBuildStep(
      `unable to build the ${host.id} CLI`,
      () => buildCliBinary(host.bunTarget),
    )
    const acnBinary = yield* localReleaseBuildStep(
      `unable to build the ${host.id} ACN`,
      () => buildAcnBinary(host.bunTarget),
    )
    const runtime = yield* packageRuntimeArtifacts(host, hostRoot, {
      cli: cliBinary,
      acn: acnBinary,
    })
    const artifacts = yield* replaceRuntimeArtifacts(manifest, runtime)
    yield* writeManifestFile(manifestPath, { ...manifest, artifacts })
    return {
      version: release.version,
      files: new Map(release.files)
        .set(runtime.cli.filename, resolve(hostRoot, cliArchive(host.id)))
        .set(runtime.acn.filename, resolve(hostRoot, acnArchive(host.id))),
    }
  })

const withStampedVersion = <A, E, R>(
  version: string,
  effect: Effect.Effect<A, E, R>,
): Effect.Effect<A, E | LocalReleaseError, R | FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const original = yield* fs.readFileString(VERSION_FILE).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`unable to read ${VERSION_FILE}: ${String(cause)}`)
      ),
    )
    const stamped = original.replace(
      /export const MAGNITUDE_VERSION = .*/,
      `export const MAGNITUDE_VERSION = ${JSON.stringify(version)}`,
    )
    if (stamped === original) {
      return yield* localReleaseFailure(
        `unable to stamp ${version} into ${VERSION_FILE}`,
      )
    }
    return yield* Effect.acquireUseRelease(
      fs.writeFileString(VERSION_FILE, stamped).pipe(
        Effect.mapError((cause) =>
          localReleaseFailure(`unable to stamp ${VERSION_FILE}: ${String(cause)}`)
        ),
      ),
      () => effect,
      () => fs.writeFileString(VERSION_FILE, original).pipe(Effect.orDie),
    )
  })

/**
 * Derives a local release at a different version from a base local release: rebuilds
 * only the version-stamped CLI and ACN binaries, and references the base's
 * version-agnostic artifacts (ICN base, backend packs) unchanged.
 */
export const deriveLocalRelease = (
  base: LocalRelease,
  version: string,
  outputRoot: string,
): Effect.Effect<LocalRelease, LocalReleaseError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const host = hostById(currentHost())
    const baseManifestPath = base.files.get("magnitude-release.json")
    if (!baseManifestPath) {
      return yield* localReleaseFailure("base local release has no manifest")
    }
    const baseManifest = yield* readManifestFile(baseManifestPath)
    yield* fs.makeDirectory(outputRoot, { recursive: true }).pipe(
      Effect.mapError((cause) =>
        localReleaseFailure(`unable to create ${outputRoot}: ${String(cause)}`)
      ),
    )

    yield* Console.log(`Deriving local release ${version} from ${base.version}`)
    const binaries = yield* withStampedVersion(version, Effect.gen(function* () {
      const cli = yield* localReleaseBuildStep(
        `unable to build the ${host.id} CLI at ${version}`,
        () => buildCliBinary(host.bunTarget),
      )
      const acn = yield* localReleaseBuildStep(
        `unable to build the ${host.id} ACN at ${version}`,
        () => buildAcnBinary(host.bunTarget),
      )
      return { cli, acn }
    }))
    const runtime = yield* packageRuntimeArtifacts(host, outputRoot, binaries)
    const artifacts = yield* replaceRuntimeArtifacts(baseManifest, runtime)
    const manifest: ReleaseManifest = {
      ...baseManifest,
      version,
      tag: releaseTag(version),
      artifacts,
    }
    const manifestPath = resolve(outputRoot, "magnitude-release.json")
    yield* writeManifestFile(manifestPath, manifest)

    const files = new Map<string, string>([
      ["magnitude-release.json", manifestPath],
    ])
    for (const artifact of manifest.artifacts) {
      if (artifact.id === runtime.cli.id || artifact.id === runtime.acn.id) {
        files.set(artifact.filename, resolve(outputRoot, artifact.filename))
        continue
      }
      const baseFile = base.files.get(artifact.filename)
      if (!baseFile) {
        return yield* localReleaseFailure(
          `base local release is missing ${artifact.filename}`,
        )
      }
      files.set(artifact.filename, baseFile)
    }
    return { version, files } satisfies LocalRelease
  })

/**
 * Loads a previously derived local release, or none when it is missing, was
 * derived from a different base (version-agnostic artifact hashes no longer
 * match), or has lost files. The caller decides whether to re-derive.
 */
export const loadDerivedLocalRelease = (
  base: LocalRelease,
  version: string,
  outputRoot: string,
): Effect.Effect<
  Option.Option<LocalRelease>,
  LocalReleaseError,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const host = hostById(currentHost())
    const baseManifestPath = base.files.get("magnitude-release.json")
    if (!baseManifestPath) {
      return yield* localReleaseFailure("base local release has no manifest")
    }
    const baseManifest = yield* readManifestFile(baseManifestPath)
    const baseArtifactsById = new Map(
      baseManifest.artifacts.map((artifact) => [artifact.id, artifact]),
    )
    const manifestPath = resolve(outputRoot, "magnitude-release.json")
    const manifest = yield* readManifestFile(manifestPath).pipe(Effect.option)
    if (Option.isNone(manifest) || manifest.value.version !== version) {
      return Option.none()
    }
    const runtimeIds = new Set([`cli-${host.id}`, `acn-${host.id}`])
    const files = new Map<string, string>([
      ["magnitude-release.json", manifestPath],
    ])
    for (const artifact of manifest.value.artifacts) {
      if (runtimeIds.has(artifact.id)) {
        const archive = resolve(outputRoot, artifact.filename)
        const exists = yield* fs.exists(archive).pipe(
          Effect.orElseSucceed(() => false),
        )
        if (!exists) return Option.none()
        files.set(artifact.filename, archive)
        continue
      }
      const baseArtifact = baseArtifactsById.get(artifact.id)
      const baseFile = base.files.get(artifact.filename)
      if (
        !baseArtifact ||
        !baseFile ||
        baseArtifact.sha256 !== artifact.sha256
      ) {
        return Option.none()
      }
      files.set(artifact.filename, baseFile)
    }
    return Option.some({ version, files } satisfies LocalRelease)
  })
