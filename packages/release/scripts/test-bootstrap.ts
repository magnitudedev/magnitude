import { createHash } from "node:crypto"
import { resolve } from "node:path"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Data, Effect, Option, Schema } from "effect"
import {
  ReleaseArtifactSchema,
  ReleaseManifestSchema,
  type ReleaseArtifact,
  type ReleaseManifest,
} from "../src/contracts"
import { releaseUrl } from "../src/acquisition"
import {
  acnArchive,
  backendPacks,
  cliArchive,
  currentHost,
  hostById,
} from "../src/targets"
import { buildAcnBinary } from "./build/acn"
import { buildBackendArtifact } from "./build/backend"
import { buildCliBinary } from "./build/cli"
import { buildHostArtifacts } from "./build/host"
import { buildArchive, run } from "./build/common"

const PROJECT_ROOT = resolve(import.meta.dir, "../../..")
const CATALOG_ROOT = resolve(PROJECT_ROOT, "inference/target/catalog-inputs")
const BUILD_ROOT = resolve(
  PROJECT_ROOT,
  "inference/target/release-bootstrap-test",
)
const RELEASE_BASE_URL = "http://127.0.0.1"
const TEST_DOWNLOAD_BYTES_PER_SECOND_PER_REQUEST = 2 * 1024 * 1024
const TEST_DOWNLOAD_CHUNK_BYTES = 64 * 1024

class BootstrapTestError extends Data.TaggedError("BootstrapTestError")<{
  readonly message: string
}> {}

interface Candidate {
  readonly version: string
  readonly files: ReadonlyMap<string, string>
}

const failure = (message: string) => new BootstrapTestError({ message })

const fromPromise = <A>(message: string, evaluate: () => Promise<A>) =>
  Effect.tryPromise({
    try: evaluate,
    catch: (cause) => failure(`${message}: ${String(cause)}`),
  })

const command = (
  executable: string,
  ...arguments_: readonly string[]
): Effect.Effect<string, BootstrapTestError> =>
  fromPromise(
    `command failed: ${[executable, ...arguments_].join(" ")}`,
    () => run([executable, ...arguments_], { cwd: PROJECT_ROOT }),
  )

const readPackageVersion = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const packageJson = yield* fs.readFileString(
    resolve(PROJECT_ROOT, "packages/cli/package.json"),
  ).pipe(
    Effect.mapError((cause) =>
      failure(`unable to read the CLI package: ${String(cause)}`)
    ),
  )
  const decoded = yield* Schema.decodeUnknown(
    Schema.parseJson(Schema.Struct({ version: Schema.NonEmptyString })),
  )(packageJson).pipe(
    Effect.mapError((cause) =>
      failure(`CLI package has an invalid version: ${String(cause)}`)
    ),
  )
  return decoded.version
})

const readArtifact = (
  descriptor: string,
): Effect.Effect<
  ReleaseArtifact,
  BootstrapTestError,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const contents = yield* fs.readFileString(descriptor).pipe(
      Effect.mapError((cause) =>
        failure(`unable to read ${descriptor}: ${String(cause)}`)
      ),
    )
    return yield* Schema.decodeUnknown(
      Schema.parseJson(ReleaseArtifactSchema),
    )(contents).pipe(
      Effect.mapError((cause) =>
        failure(
          `invalid artifact descriptor ${descriptor}: ${String(cause)}`,
        )
      ),
    )
  })

const buildCandidate = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const host = currentHost()
  const root = resolve(BUILD_ROOT, host)
  const hostRoot = resolve(root, "host")
  const packs = backendPacks.filter(
    (pack) => pack.host === host && pack.backend === "metal",
  )

  yield* Console.log("Hydrating the release catalog")
  yield* command("bun", "run", "icn:catalog:hydrate")

  yield* Console.log(`Building ${host} CLI, ACN, and ICN release artifacts`)
  yield* fromPromise(
    `unable to build ${host} release artifacts`,
    () => buildHostArtifacts(host, CATALOG_ROOT, hostRoot),
  )

  for (const pack of packs) {
    const packRoot = resolve(root, pack.id)
    yield* Console.log(`Building ${pack.id} release artifact`)
    yield* fromPromise(
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
          failure(`unable to inspect ${artifactRoot}: ${String(cause)}`)
        ),
      )
    )
  ).flat()
  const artifacts = yield* Effect.forEach(descriptors, readArtifact)
  const byId = new Map(artifacts.map((artifact) => [artifact.id, artifact]))
  const base = byId.get(`icn-base-${host}`)
  if (!base) return yield* failure(`candidate has no ICN base for ${host}`)
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
      return yield* failure(`${pack.id} is incompatible with ${base.id}`)
    }
  }

  const version = yield* readPackageVersion
  const sourceCommit = (yield* command(
    "git",
    "rev-parse",
    "HEAD",
  )).trim()
  const sortedArtifacts = artifacts
    .slice()
    .sort((left, right) => left.id.localeCompare(right.id))
  const [firstArtifact, ...remainingArtifacts] = sortedArtifacts
  if (!firstArtifact) {
    return yield* failure("candidate has no release artifacts")
  }
  const manifest = {
    schemaVersion: 1,
    version,
    tag: `@magnitudedev/cli@${version}`,
    sourceCommit,
    artifacts: [firstArtifact, ...remainingArtifacts],
  } satisfies ReleaseManifest
  const serialized = yield* Schema.encode(
    Schema.parseJson(ReleaseManifestSchema),
  )(manifest).pipe(
    Effect.mapError((cause) =>
      failure(
        `unable to encode the local release manifest: ${String(cause)}`,
      )
    ),
  )
  const manifestPath = resolve(root, "magnitude-release.json")
  yield* fs.writeFileString(manifestPath, `${serialized}\n`).pipe(
    Effect.mapError((cause) =>
      failure(
        `unable to write the local release manifest: ${String(cause)}`,
      )
    ),
  )

  const files = new Map<string, string>([
    ["magnitude-release.json", manifestPath],
  ])
  for (const artifact of artifacts) {
    const artifactRoot = artifactRoots.find((candidate) =>
      descriptors.includes(resolve(
        candidate,
        `${artifact.id}.artifact.json`,
      ))
    )
    if (!artifactRoot) {
      return yield* failure(`unable to locate ${artifact.id}`)
    }
    files.set(artifact.filename, resolve(artifactRoot, artifact.filename))
  }
  return { version, files } satisfies Candidate
})

const loadCandidate = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const host = currentHost()
  const root = resolve(BUILD_ROOT, host)
  const version = yield* readPackageVersion
  const manifestPath = resolve(root, "magnitude-release.json")
  const contents = yield* fs.readFileString(manifestPath).pipe(
    Effect.mapError((cause) =>
      failure(`no local bootstrap candidate exists: ${String(cause)}`)
    ),
  )
  const manifest = yield* Schema.decodeUnknown(
    Schema.parseJson(ReleaseManifestSchema),
  )(contents).pipe(
    Effect.mapError((cause) =>
      failure(
        `reusable bootstrap candidate has an invalid manifest: ${String(cause)}`,
      )
    ),
  )
  if (manifest.version !== version) {
    return yield* failure(
      `reusable candidate is ${manifest.version}, but the CLI is ${version}`,
    )
  }
  const files = new Map<string, string>([
    ["magnitude-release.json", manifestPath],
  ])
  for (const artifact of manifest.artifacts) {
    const candidates = [
      resolve(root, "host", artifact.filename),
      ...backendPacks
        .filter((pack) => pack.host === host)
        .map((pack) => resolve(root, pack.id, artifact.filename)),
    ]
    const existing = yield* Effect.findFirst(candidates, (candidate) =>
      fs.exists(candidate)
    )
    if (Option.isNone(existing)) {
      return yield* failure(`reusable candidate is missing ${artifact.filename}`)
    }
    files.set(artifact.filename, existing.value)
  }
  return { version, files } satisfies Candidate
})

const refreshRuntimeCandidate = (
  candidate: Candidate,
): Effect.Effect<
  Candidate,
  BootstrapTestError,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const host = hostById(currentHost())
    const root = resolve(BUILD_ROOT, host.id)
    const hostRoot = resolve(root, "host")
    const manifestPath = resolve(root, "magnitude-release.json")
    const manifestContents = yield* fs.readFileString(manifestPath).pipe(
      Effect.mapError((cause) =>
        failure(
          `unable to read the reusable release manifest: ${String(cause)}`,
        )
      ),
    )
    const manifest = yield* Schema.decodeUnknown(
      Schema.parseJson(ReleaseManifestSchema),
    )(manifestContents).pipe(
      Effect.mapError((cause) =>
        failure(
          `reusable release manifest is invalid: ${String(cause)}`,
        )
      ),
    )

    yield* command(
      "bun",
      "run",
      "packages/version/scripts/generate-version.ts",
    )
    yield* Console.log(`Refreshing the ${host.id} CLI and ACN release artifacts`)
    const cliBinary = yield* fromPromise(
      `unable to build the ${host.id} CLI`,
      () => buildCliBinary(host.bunTarget),
    )
    const acnBinary = yield* fromPromise(
      `unable to build the ${host.id} ACN`,
      () => buildAcnBinary(host.bunTarget),
    )
    const cliArchivePath = resolve(hostRoot, cliArchive(host.id))
    const cliDescriptorPath = resolve(
      hostRoot,
      `cli-${host.id}.artifact.json`,
    )
    const acnArchivePath = resolve(hostRoot, acnArchive(host.id))
    const acnDescriptorPath = resolve(
      hostRoot,
      `acn-${host.id}.artifact.json`,
    )
    yield* Effect.all(
      [
        cliArchivePath,
        cliDescriptorPath,
        acnArchivePath,
        acnDescriptorPath,
      ].map((path) =>
        fs.remove(path, { force: true }).pipe(
          Effect.mapError((cause) =>
            failure(`unable to replace ${path}: ${String(cause)}`)
          ),
        )
      ),
      { discard: true },
    )
    const cliArtifact = yield* fromPromise(
      `unable to package the ${host.id} CLI`,
      () =>
        buildArchive(
          cliArchivePath,
          cliDescriptorPath,
          {
            id: `cli-${host.id}`,
            kind: "cli",
            host: Option.some(host.id),
            backend: Option.none(),
            requiredBaseId: Option.none(),
            nativeBuild: Option.none(),
            backendModuleAbi: Option.none(),
            compatibility: Option.none(),
          },
          [{
            path: `bin/magnitude-cli${host.executableExtension}`,
            source: cliBinary,
            mode: 0o755,
          }],
        ),
    )
    const acnArtifact = yield* fromPromise(
      `unable to package the ${host.id} ACN`,
      () =>
        buildArchive(
          acnArchivePath,
          acnDescriptorPath,
          {
            id: `acn-${host.id}`,
            kind: "acn",
            host: Option.some(host.id),
            backend: Option.none(),
            requiredBaseId: Option.none(),
            nativeBuild: Option.none(),
            backendModuleAbi: Option.none(),
            compatibility: Option.none(),
          },
          [{
            path: `bin/magnitude-acn${host.executableExtension}`,
            source: acnBinary,
            mode: 0o755,
          }],
        ),
    )
    const updatedArtifacts = manifest.artifacts.map((current) =>
      current.id === cliArtifact.id
        ? cliArtifact
        : current.id === acnArtifact.id
          ? acnArtifact
          : current
    )
    const [firstArtifact, ...remainingArtifacts] = updatedArtifacts
    if (!firstArtifact) return yield* failure("candidate has no artifacts")
    const updatedManifest: ReleaseManifest = {
      ...manifest,
      artifacts: [firstArtifact, ...remainingArtifacts],
    }
    const serialized = yield* Schema.encode(
      Schema.parseJson(ReleaseManifestSchema),
    )(updatedManifest).pipe(
      Effect.mapError((cause) =>
        failure(
          `unable to encode the refreshed release manifest: ${String(cause)}`,
        )
      ),
    )
    yield* fs.writeFileString(manifestPath, `${serialized}\n`).pipe(
      Effect.mapError((cause) =>
        failure(
          `unable to publish the refreshed release manifest: ${String(cause)}`,
        )
      ),
    )
    return {
      version: candidate.version,
      files: new Map(candidate.files)
        .set(cliArtifact.filename, cliArchivePath)
        .set(acnArtifact.filename, acnArchivePath),
    }
  })

const parseRange = (
  value: string,
  size: number,
): Option.Option<{ readonly start: number; readonly end: number }> => {
  const match = /^bytes=(\d+)-(\d*)$/.exec(value)
  if (!match) return Option.none()
  const start = Number(match[1])
  const requestedEnd = match[2] === "" ? size - 1 : Number(match[2])
  const end = Math.min(requestedEnd, size - 1)
  return Number.isSafeInteger(start) &&
      Number.isSafeInteger(end) &&
      start >= 0 &&
      start <= end
    ? Option.some({ start, end })
    : Option.none()
}

const throttled = (blob: Blob): ReadableStream<Uint8Array> => {
  let offset = 0
  return new ReadableStream({
    async pull(controller) {
      if (offset >= blob.size) {
        controller.close()
        return
      }
      const end = Math.min(offset + TEST_DOWNLOAD_CHUNK_BYTES, blob.size)
      const chunk = new Uint8Array(
        await blob.slice(offset, end).arrayBuffer(),
      )
      offset = end
      controller.enqueue(chunk)
      await Bun.sleep(
        chunk.byteLength /
          TEST_DOWNLOAD_BYTES_PER_SECOND_PER_REQUEST *
          1_000,
      )
    },
  })
}

const serveCandidate = (candidate: Candidate) =>
  Effect.acquireRelease(
    Effect.sync(() => {
      const routes = new Map(
        [...candidate.files].map(([name, path]) => [
          new URL(
            releaseUrl(RELEASE_BASE_URL, candidate.version, name),
          ).pathname,
          { name, path },
        ]),
      )
      return Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        idleTimeout: 60,
        async fetch(request) {
          const route = routes.get(new URL(request.url).pathname)
          if (!route) return new Response("Not found", { status: 404 })
          const { name, path } = route
          const file = Bun.file(path)
          if (!(await file.exists())) {
            return new Response("Not found", { status: 404 })
          }
          const throttle =
            name.startsWith("magnitude-acn-") ||
            name.startsWith("magnitude-icn-")
          const body = (blob: Blob): Blob | ReadableStream<Uint8Array> =>
            throttle ? throttled(blob) : blob
          const etag = `"${createHash("sha256")
            .update(`${path}:${file.size}`)
            .digest("hex")}"`
          const commonHeaders = {
            "accept-ranges": "bytes",
            etag,
          }
          if (request.method === "HEAD") {
            return new Response(null, {
              headers: {
                ...commonHeaders,
                "content-length": String(file.size),
              },
            })
          }
          if (request.method !== "GET") {
            return new Response("Method not allowed", { status: 405 })
          }
          const requestedRange = Option.fromNullable(
            request.headers.get("range"),
          )
          if (Option.isSome(requestedRange)) {
            const ifRange = Option.fromNullable(request.headers.get("if-range"))
            if (Option.isSome(ifRange) && ifRange.value !== etag) {
              return new Response(body(file), {
                headers: {
                  ...commonHeaders,
                  "content-length": String(file.size),
                },
              })
            }
            const range = parseRange(requestedRange.value, file.size)
            if (Option.isNone(range)) {
              return new Response("Range not satisfiable", {
                status: 416,
                headers: {
                  ...commonHeaders,
                  "content-range": `bytes */${file.size}`,
                },
              })
            }
            const { start, end } = range.value
            return new Response(body(file.slice(start, end + 1)), {
              status: 206,
              headers: {
                ...commonHeaders,
                "content-length": String(end - start + 1),
                "content-range": `bytes ${start}-${end}/${file.size}`,
              },
            })
          }
          return new Response(body(file), {
            headers: {
              ...commonHeaders,
              "content-length": String(file.size),
            },
          })
        },
      })
    }),
    (server) => Effect.sync(() => server.stop(true)),
  )

const launchCli = (
  candidate: Candidate,
  baseUrl: string,
  home: string,
  arguments_: readonly string[],
): Effect.Effect<number, BootstrapTestError> =>
  Effect.async((resume) => {
    const excluded = new Set([
      "HOME",
      "USERPROFILE",
      "MAGNITUDE_ACN_VERSION",
      "MAGNITUDE_ICN_PATH",
      "MAGNITUDE_RELEASE_BASE_URL",
      "MAGNITUDE_USE_LOCAL",
    ])
    const inherited: Record<string, string> = {}
    for (const [name, value] of Object.entries(process.env)) {
      const present = Option.fromNullable(value)
      if (!excluded.has(name) && Option.isSome(present)) {
        inherited[name] = present.value
      }
    }
    const child = Bun.spawn(
      [
        "node",
        resolve(PROJECT_ROOT, "packages/cli/bin/magnitude.js"),
        ...arguments_,
      ],
      {
        cwd: PROJECT_ROOT,
        env: {
          ...inherited,
          HOME: home,
          USERPROFILE: home,
          MAGNITUDE_RELEASE_BASE_URL: baseUrl,
        },
        stdin: "inherit",
        stdout: "inherit",
        stderr: "inherit",
      },
    )
    child.exited.then(
      (code) => resume(Effect.succeed(code)),
      (cause) =>
        resume(Effect.fail(
          failure(`unable to run the CLI: ${String(cause)}`),
        )),
    )
    return Effect.sync(() => child.kill("SIGTERM"))
  })

const usage = `Usage: bun test:release-bootstrap [--rebuild] [-- <CLI arguments>]

Build and run the current worktree through the production release acquisition path.

  --rebuild  Rebuild the local release candidate before running.
  --help     Show this help.
`

const program = Effect.gen(function* () {
  const arguments_ = process.argv.slice(2)
  if (arguments_.includes("--help")) {
    yield* Console.log(usage)
    return
  }
  const rebuild = arguments_.includes("--rebuild")
  const cliArguments = arguments_.filter(
    (argument) => argument !== "--rebuild" && argument !== "--",
  )

  const loadedCandidate = rebuild
    ? yield* buildCandidate
    : yield* loadCandidate.pipe(
      Effect.catchTag("BootstrapTestError", () => buildCandidate),
    )
  const candidate = rebuild
    ? loadedCandidate
    : yield* refreshRuntimeCandidate(loadedCandidate)
  yield* command(
    "bun",
    "run",
    "packages/cli/scripts/prepare-release-runtime.ts",
  )

  const fs = yield* FileSystem.FileSystem
  const home = yield* fs.makeTempDirectory({
    prefix: "magnitude-bootstrap-test-",
  }).pipe(
    Effect.mapError((cause) =>
      failure(
        `unable to create an isolated Magnitude home: ${String(cause)}`,
      )
    ),
  )
  yield* Effect.scoped(
    Effect.gen(function* () {
      const server = yield* serveCandidate(candidate)
      const baseUrl = `http://127.0.0.1:${server.port}`
      yield* Console.log([
        "",
        `Local release: ${candidate.version}`,
        `Isolated home: ${home}`,
        `Release server: ${baseUrl}`,
        "",
        "Starting a cold production-style bootstrap...",
        "",
      ].join("\n"))
      const exitCode = yield* launchCli(
        candidate,
        baseUrl,
        home,
        cliArguments,
      )
      yield* Console.log(`\nBootstrap state preserved at ${home}`)
      if (exitCode !== 0) {
        return yield* failure(`Magnitude exited with code ${exitCode}`)
      }
    }),
  )
}).pipe(
  Effect.catchTag("BootstrapTestError", (error) =>
    Console.error(`Release bootstrap test failed: ${error.message}`).pipe(
      Effect.zipRight(Effect.sync(() => {
        process.exitCode = 1
      })),
    )
  ),
  Effect.provide(BunContext.layer),
)

BunRuntime.runMain(program)
