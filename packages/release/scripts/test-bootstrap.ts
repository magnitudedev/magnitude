import { createHash } from "node:crypto"
import { resolve } from "node:path"
import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { FetchHttpClient } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Data, Effect, Option } from "effect"
import { resolveReleaseIcnInstallation } from "../../icn/src/lifecycle/release-installation"
import { IcnPreparationReporter } from "../../icn/src/lifecycle/preparation"
import { smokeInstallation } from "../../../inference/scripts/smoke"
import { releaseUrl } from "../src/acquisition"
import { currentHost, desktopUpdateArchive } from "../src/targets"
import {
  buildLocalRelease,
  loadLocalRelease,
  type LocalRelease,
} from "./local-release"

const RELEASE_BASE_URL = "http://127.0.0.1"
const TEST_DOWNLOAD_BYTES_PER_SECOND_PER_REQUEST = 2 * 1024 * 1024
const TEST_DOWNLOAD_CHUNK_BYTES = 64 * 1024

class BootstrapTestError extends Data.TaggedError("BootstrapTestError")<{
  readonly message: string
}> {}

const failure = (message: string) => new BootstrapTestError({ message })

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

const serveLocalRelease = (release: LocalRelease) =>
  Effect.acquireRelease(
    Effect.sync(() => {
      const routes = new Map(
        [...release.files].map(([name, path]) => [
          new URL(
            releaseUrl(RELEASE_BASE_URL, release.version, name),
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

const usage = `Usage: bun test:release-bootstrap [--cached]

Build this host's release and acquire its engine into an empty profile through the production installation path. Verify the installed engine's identity, readiness, authenticated hardware, and parent-loss shutdown.

  --cached  Reuse the last complete local release for harness iteration; this does not validate current source changes.
  --help     Show this help.
`

const program = Effect.gen(function* () {
  const arguments_ = process.argv.slice(2)
  if (arguments_.includes("--help")) {
    yield* Console.log(usage)
    return
  }
  const cached = arguments_.includes("--cached")
  const unexpected = arguments_.filter((argument) => argument !== "--cached")
  if (unexpected.length > 0) {
    return yield* failure(`unsupported arguments: ${unexpected.join(" ")}`)
  }

  const release = cached ? yield* loadLocalRelease : yield* buildLocalRelease

  const fs = yield* FileSystem.FileSystem
  const home = yield* fs.makeTempDirectory({
    prefix: "magnitude-bootstrap-test-",
    // The application's Unix control socket must fit macOS's 103-byte path limit.
    ...(process.platform === "darwin" ? { directory: "/tmp" } : {}),
  }).pipe(
    Effect.mapError((cause) =>
      failure(
        `unable to create an isolated Magnitude home: ${String(cause)}`,
      )
    ),
  )
  yield* Effect.scoped(
    Effect.gen(function* () {
      const server = yield* serveLocalRelease(release)
      const baseUrl = `http://127.0.0.1:${server.port}`
      yield* Console.log([
        "",
        `Local release: ${release.version}`,
        `Artifact source: ${cached ? "cached local release (source edits unverified)" : "current worktree build"}`,
        `Isolated home: ${home}`,
        `Release server: ${baseUrl}`,
        "",
        "Downloading and installing the release engine into the empty profile...",
        "",
      ].join("\n"))
      const installation = yield* resolveReleaseIcnInstallation(
        release.version,
        resolve(home, ".magnitude"),
        baseUrl,
      ).pipe(
        Effect.provideService(IcnPreparationReporter, { report: () => Effect.void }),
        Effect.provide(FetchHttpClient.layer),
        Effect.mapError((error) => failure(`engine acquisition failed at ${error.stage}: ${error.message}`)),
      )
      yield* Effect.tryPromise({
        try: () => smokeInstallation(installation.declarationPath),
        catch: (cause) => failure(`installed engine smoke failed: ${String(cause)}`),
      })
      yield* Console.log(`Installed engine ready: ${installation.declarationPath}`)
      const host = currentHost()
      if (host === "darwin-arm64" || host === "darwin-x64") {
        const archive = release.files.get(desktopUpdateArchive(host))
        if (!archive) return yield* failure("local release has no desktop update archive")
        const desktopRoot = resolve(home, "desktop")
        yield* fs.makeDirectory(desktopRoot)
        const extracted = yield* Command.make("/usr/bin/ditto", "-x", "-k", archive, desktopRoot).pipe(Command.exitCode)
        if (extracted !== 0) return yield* failure(`desktop update archive extraction exited ${extracted}`)
        const resources = resolve(desktopRoot, "Magnitude.app/Contents/Resources")
        const acceptanceRoot = yield* fs.makeTempDirectory({ prefix: "mag-headless-", directory: "/tmp" })
        for (const [offline, endpoint] of [[false, baseUrl], [true, "http://127.0.0.1:1"]] as const) {
          const acceptance = yield* Command.make(process.execPath, "run", resolve(import.meta.dir, "acceptance/test-installed-headless.ts")).pipe(
            Command.env({
              MAGNITUDE_INSTALLED_ACCEPTANCE_OUTPUT: acceptanceRoot,
              MAGNITUDE_INSTALLED_ACCEPTANCE_CLI: resolve(resources, "magnitude"),
              MAGNITUDE_INSTALLED_ACCEPTANCE_ADDON: resolve(resources, "desktop-host.node"),
              MAGNITUDE_INSTALLED_ACCEPTANCE_VERSION: release.version,
              MAGNITUDE_INSTALLED_ACCEPTANCE_OFFLINE: String(offline),
              MAGNITUDE_RELEASE_BASE_URL: endpoint,
              MAGNITUDE_ICN_PATH: "",
            }),
            Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
            Effect.timeoutFail({ duration: "2 minutes", onTimeout: () => failure("installed headless acceptance timed out") }),
          )
          if (acceptance !== 0) return yield* failure(`installed headless acceptance exited ${acceptance}`)
        }
        yield* Console.log(`Installed desktop and bundled CLI acquired the engine and restarted offline: ${acceptanceRoot}`)
      }
      yield* Console.log(`\nBootstrap state preserved at ${home}`)
    }),
  )
}).pipe(
  Effect.catchTags({
    BootstrapTestError: (error: BootstrapTestError) =>
      Console.error(`Release bootstrap test failed: ${error.message}`).pipe(
        Effect.zipRight(Effect.sync(() => {
          process.exitCode = 1
        })),
      ),
    LocalReleaseError: (error) =>
      Console.error(`Release bootstrap test failed: ${error.message}`).pipe(
        Effect.zipRight(Effect.sync(() => {
          process.exitCode = 1
        })),
      ),
  }),
)

BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
