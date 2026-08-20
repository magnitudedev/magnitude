import { createHash } from "node:crypto"
import { resolve } from "node:path"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Data, Effect, Option } from "effect"
import { releaseUrl } from "../src/acquisition"
import {
  buildLocalRelease,
  loadLocalRelease,
  refreshLocalRelease,
  runRepoCommand,
  type LocalRelease,
} from "./local-release"

const PROJECT_ROOT = resolve(import.meta.dir, "../../..")
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

const launchCli = (
  release: LocalRelease,
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
        resolve(PROJECT_ROOT, "packages/launcher/bin/magnitude.js"),
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

  --rebuild  Rebuild the local release before running.
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

  const loadedRelease = rebuild
    ? yield* buildLocalRelease
    : yield* loadLocalRelease.pipe(
      Effect.catchTag("LocalReleaseError", () => buildLocalRelease),
    )
  const release = rebuild
    ? loadedRelease
    : yield* refreshLocalRelease(loadedRelease)
  yield* runRepoCommand(
    "bun",
    "run",
    "packages/launcher/scripts/build-launcher.ts",
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
      const server = yield* serveLocalRelease(release)
      const baseUrl = `http://127.0.0.1:${server.port}`
      yield* Console.log([
        "",
        `Local release: ${release.version}`,
        `Isolated home: ${home}`,
        `Release server: ${baseUrl}`,
        "",
        "Starting a cold production-style bootstrap...",
        "",
      ].join("\n"))
      const exitCode = yield* launchCli(
        release,
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
