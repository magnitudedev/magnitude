import { createHash } from "node:crypto"
import { spawn, type ChildProcess } from "node:child_process"
import {
  chmod,
  cp,
  mkdir,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises"
import { resolve } from "node:path"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Data, Effect, Option } from "effect"
import { releaseTag } from "../src/contracts"
import { run } from "./build/common"
import {
  buildLocalRelease,
  deriveLocalRelease,
  loadDerivedLocalRelease,
  loadLocalRelease,
  runRepoCommand,
  type LocalRelease,
} from "./local-release"

/**
 * Simulates the distribution around the real product: one local server stands
 * in for both remote hosts (the npm registry and the GitHub release
 * downloads), serving two local releases. Everything else — launcher, native
 * CLI binaries, npm itself, acquisition, the updater — is real. You drive the
 * install → publish → update-prompt → upgrade → relaunch lifecycle by hand in
 * an isolated subshell.
 */
const INSTALLED_VERSION = "0.0.1"
const PUBLISHED_VERSION = "0.0.2"

const PROJECT_ROOT = resolve(import.meta.dir, "../../..")
const SIMULATION_ROOT = resolve(
  PROJECT_ROOT,
  "inference/target/release-simulation",
)

class SimulationError extends Data.TaggedError("SimulationError")<{
  readonly message: string
}> {}

const simulationFailure = (message: string) => new SimulationError({ message })

const simulationStep = <A>(message: string, evaluate: () => Promise<A>) =>
  Effect.tryPromise({
    try: evaluate,
    catch: (cause) => simulationFailure(`${message}: ${String(cause)}`),
  })

// The server runs in a node child: it must stay reachable from npm and the
// launcher (both separate processes), and the port travels through a file so
// nothing depends on child stdio pipes.
const STATIC_SERVER_SOURCE = `
const { createServer } = require("node:http")
const { createReadStream, existsSync, statSync, writeFileSync } = require("node:fs")
const { join } = require("node:path")
const root = process.argv[1]
const portPath = process.argv[2]
const server = createServer((request, response) => {
  const segments = new URL(request.url, "http://distribution.simulated").pathname
    .split("/").filter(Boolean).map(decodeURIComponent)
  const target = join(root, ...segments)
  if (!target.startsWith(root) || !existsSync(target) || !statSync(target).isFile()) {
    response.statusCode = 404
    response.end("not found")
    return
  }
  const name = segments.at(-1) ?? ""
  response.setHeader("content-type", !name.includes(".") || name.endsWith(".json")
    ? "application/json"
    : "application/octet-stream")
  response.setHeader("content-length", statSync(target).size)
  if (request.method === "HEAD") {
    response.end()
    return
  }
  createReadStream(target).pipe(response)
})
server.listen(0, "127.0.0.1", () => writeFileSync(portPath, String(server.address().port)))
`

interface SessionPaths {
  readonly session: string
  readonly serve: string
  readonly home: string
  readonly prefix: string
  readonly npmCache: string
  readonly npmUserconfig: string
  readonly portFile: string
  readonly publishScript: string
}

const sessionPaths = (): SessionPaths => {
  const session = resolve(SIMULATION_ROOT, "session")
  return {
    session,
    serve: resolve(session, "serve"),
    home: resolve(session, "home"),
    prefix: resolve(session, "prefix"),
    npmCache: resolve(session, "npm-cache"),
    npmUserconfig: resolve(session, "npmrc"),
    portFile: resolve(session, "port"),
    publishScript: resolve(session, `publish-${PUBLISHED_VERSION}.sh`),
  }
}

/** Lays a local release out under the paths the release downloader requests. */
const stageLocalRelease = (serve: string, release: LocalRelease) =>
  simulationStep(`unable to stage local release ${release.version}`, async () => {
    const tagDirectory = resolve(serve, ...releaseTag(release.version).split("/"))
    await mkdir(tagDirectory, { recursive: true })
    for (const [name, path] of release.files) {
      await symlink(path, resolve(tagDirectory, name))
    }
  })

const packLauncherTarball = (
  paths: SessionPaths,
  launcherArtifact: string,
  version: string,
) =>
  simulationStep(`unable to pack the ${version} npm tarball`, async () => {
    const staging = resolve(paths.session, `pack-${version}`)
    const packageDirectory = resolve(staging, "package")
    await mkdir(resolve(packageDirectory, "bin"), { recursive: true })
    await writeFile(resolve(packageDirectory, "package.json"), JSON.stringify({
      name: "@magnitudedev/cli",
      version,
      bin: { magnitude: "bin/magnitude.js" },
      files: ["bin/magnitude.js"],
    }))
    await cp(launcherArtifact, resolve(packageDirectory, "bin", "magnitude.js"))
    await chmod(resolve(packageDirectory, "bin", "magnitude.js"), 0o755)
    const tarball = resolve(staging, "cli.tgz")
    await run(["tar", "-czf", tarball, "-C", staging, "package"])
    const bytes = await readFile(tarball)
    const destination = resolve(paths.serve, "tarballs", `cli-${version}.tgz`)
    await mkdir(resolve(paths.serve, "tarballs"), { recursive: true })
    await cp(tarball, destination)
    return bytes
  })

const packumentVersion = (baseUrl: string, version: string, tarball: Buffer) => ({
  name: "@magnitudedev/cli",
  version,
  bin: { magnitude: "bin/magnitude.js" },
  dist: {
    tarball: `${baseUrl}/tarballs/cli-${version}.tgz`,
    shasum: createHash("sha1").update(tarball).digest("hex"),
    integrity: `sha512-${createHash("sha512").update(tarball).digest("base64")}`,
  },
})

/**
 * Writes the registry state for a given `latest`, plus the pre-staged files
 * and script that flip it — the simulated "the team published a release"
 * event.
 */
const writeRegistry = (
  paths: SessionPaths,
  baseUrl: string,
  tarballs: { readonly [version: string]: Buffer },
) =>
  simulationStep("unable to write the simulated registry", async () => {
    const packumentFor = (latest: string) => JSON.stringify({
      name: "@magnitudedev/cli",
      "dist-tags": { latest },
      versions: Object.fromEntries(
        Object.entries(tarballs).map(([version, bytes]) => [
          version,
          packumentVersion(baseUrl, version, bytes),
        ]),
      ),
    })
    const distTagsFor = (latest: string) => JSON.stringify({ latest })

    const packumentPath = resolve(paths.serve, "@magnitudedev", "cli")
    const distTagsPath = resolve(paths.serve, "registry", "dist-tags.json")
    await mkdir(resolve(paths.serve, "@magnitudedev"), { recursive: true })
    await mkdir(resolve(paths.serve, "registry"), { recursive: true })
    await writeFile(packumentPath, packumentFor(INSTALLED_VERSION))
    await writeFile(distTagsPath, distTagsFor(INSTALLED_VERSION))

    const publishedPackument = resolve(
      paths.session,
      `packument-latest-${PUBLISHED_VERSION}.json`,
    )
    const publishedDistTags = resolve(
      paths.session,
      `dist-tags-latest-${PUBLISHED_VERSION}.json`,
    )
    await writeFile(publishedPackument, packumentFor(PUBLISHED_VERSION))
    await writeFile(publishedDistTags, distTagsFor(PUBLISHED_VERSION))
    const updateCache = resolve(paths.home, ".magnitude", "version.json")
    await writeFile(paths.publishScript, [
      "#!/bin/sh",
      `cp ${JSON.stringify(publishedPackument)} ${JSON.stringify(packumentPath)}`,
      `cp ${JSON.stringify(publishedDistTags)} ${JSON.stringify(distTagsPath)}`,
      `echo "Published ${PUBLISHED_VERSION}: the registry's latest now points at it."`,
      // Publishing compresses days into an instant, so the simulated time skip
      // must age the update cache too — discovery only re-runs when the last
      // check is older than its TTL.
      `if [ -f ${JSON.stringify(updateCache)} ]; then`,
      `  node -e 'const path = process.argv[1]; const fs = require("node:fs"); const cache = JSON.parse(fs.readFileSync(path, "utf8")); cache.lastCheckedAt = new Date(0).toISOString(); fs.writeFileSync(path, JSON.stringify(cache))' ${JSON.stringify(updateCache)}`,
      `  echo "Aged the update cache (simulates the days since the last check)."`,
      "fi",
      "",
    ].join("\n"))
    await chmod(paths.publishScript, 0o755)
  })

const startServer = (paths: SessionPaths) =>
  Effect.acquireRelease(
    Effect.gen(function* () {
      const child: ChildProcess = spawn(
        "node",
        ["-e", STATIC_SERVER_SOURCE, paths.serve, paths.portFile],
      )
      const port = yield* simulationStep(
        "the distribution server did not start",
        () => new Promise<string>((resolvePort, reject) => {
          child.once("error", reject)
          child.once("exit", (code) =>
            reject(new Error(`server exited with ${code}`)))
          const poll = setInterval(() => {
            readFile(paths.portFile, "utf8").then((value) => {
              clearInterval(poll)
              resolvePort(value.trim())
            }, () => {})
          }, 25)
        }),
      )
      return { child, baseUrl: `http://127.0.0.1:${port}` }
    }),
    (server) => Effect.sync(() => { server.child.kill() }),
  )

const subshellEnvironment = (
  paths: SessionPaths,
  baseUrl: string,
): Record<string, string> => {
  const inherited: Record<string, string> = {}
  for (const [name, value] of Object.entries(process.env)) {
    if (value !== undefined) inherited[name] = value
  }
  return {
    ...inherited,
    PATH: `${resolve(paths.prefix, "bin")}:${inherited.PATH ?? ""}`,
    HOME: paths.home,
    USERPROFILE: paths.home,
    npm_config_registry: baseUrl,
    npm_config_prefix: paths.prefix,
    npm_config_cache: paths.npmCache,
    npm_config_userconfig: paths.npmUserconfig,
    npm_config_audit: "false",
    npm_config_fund: "false",
    npm_config_update_notifier: "false",
    MAGNITUDE_RELEASE_BASE_URL: baseUrl,
    MAGNITUDE_NPM_PACKAGE_URL: `${baseUrl}/registry/dist-tags.json`,
  }
}

const usage = `Usage: bun simulate:distribution [--reuse] [--rebuild]

Simulate the distribution (npm registry + release downloads) around the real
product, and drive install and update by hand in an isolated subshell.

  --reuse    Reuse previously derived ${INSTALLED_VERSION}/${PUBLISHED_VERSION} local releases
             instead of rebuilding their CLI/ACN binaries from current code.
  --rebuild  Rebuild the base local release (including ICN) from scratch first.
  --help     Show this help.
`

const program = Effect.gen(function* () {
  const arguments_ = process.argv.slice(2)
  if (arguments_.includes("--help")) {
    yield* Console.log(usage)
    return
  }
  const reuse = arguments_.includes("--reuse")
  const rebuild = arguments_.includes("--rebuild")

  const base = rebuild
    ? yield* buildLocalRelease
    : yield* loadLocalRelease.pipe(
      Effect.catchTag("LocalReleaseError", () => buildLocalRelease),
    )

  const localReleaseFor = (version: string) =>
    Effect.gen(function* () {
      const outputRoot = resolve(SIMULATION_ROOT, version)
      if (reuse) {
        const cached = yield* loadDerivedLocalRelease(base, version, outputRoot)
        if (Option.isSome(cached)) return cached.value
      }
      return yield* deriveLocalRelease(base, version, outputRoot)
    })
  const installedRelease = yield* localReleaseFor(INSTALLED_VERSION)
  const publishedRelease = yield* localReleaseFor(PUBLISHED_VERSION)

  yield* runRepoCommand("bun", "run", "packages/launcher/scripts/build-launcher.ts")
  const launcherArtifact = resolve(PROJECT_ROOT, "packages/launcher/bin/magnitude.js")

  const paths = sessionPaths()
  yield* simulationStep("unable to reset the simulation session", async () => {
    await rm(paths.session, { recursive: true, force: true })
    for (const directory of [paths.serve, paths.home, paths.prefix, paths.npmCache]) {
      await mkdir(directory, { recursive: true })
    }
    await writeFile(paths.npmUserconfig, "")
  })
  yield* stageLocalRelease(paths.serve, installedRelease)
  yield* stageLocalRelease(paths.serve, publishedRelease)

  yield* Effect.scoped(Effect.gen(function* () {
    const server = yield* startServer(paths)
    const tarballs = {
      [INSTALLED_VERSION]: yield* packLauncherTarball(
        paths,
        launcherArtifact,
        INSTALLED_VERSION,
      ),
      [PUBLISHED_VERSION]: yield* packLauncherTarball(
        paths,
        launcherArtifact,
        PUBLISHED_VERSION,
      ),
    }
    yield* writeRegistry(paths, server.baseUrl, tarballs)

    const environment = subshellEnvironment(paths, server.baseUrl)
    yield* Console.log(`Installing @magnitudedev/cli@${INSTALLED_VERSION} with npm...`)
    yield* simulationStep("npm install failed", () =>
      run(
        ["npm", "install", "-g", `@magnitudedev/cli@${INSTALLED_VERSION}`],
        { env: environment },
      ))

    yield* Console.log([
      "",
      "Distribution simulation ready.",
      `  Simulated registry + releases: ${server.baseUrl}`,
      `  Installed: @magnitudedev/cli@${INSTALLED_VERSION} (registry latest: ${INSTALLED_VERSION})`,
      `  Staged but unpublished: ${PUBLISHED_VERSION}`,
      `  Isolated HOME: ${paths.home}`,
      "",
      "Suggested flow:",
      "  magnitude                     cold install and run of the native CLI",
      `  ${paths.publishScript}`,
      `                                "publish" ${PUBLISHED_VERSION}: flips the registry's latest and ages`,
      "                                the update cache (the simulated time skip)",
      "  magnitude                     background update discovery finds the new version",
      `  magnitude                     the update prompt offers ${INSTALLED_VERSION} -> ${PUBLISHED_VERSION}`,
      "  magnitude                     after updating: you are on the new version",
      "",
      "The subshell's HOME and npm configuration are isolated; Ctrl+D to end",
      "the session (the server stops and the simulated world is removed;",
      "derived local releases stay cached).",
      "",
    ].join("\n"))

    const shell = process.env.SHELL ?? "/bin/sh"
    yield* simulationStep("the simulation subshell failed to run", () => {
      const child = spawn(shell, {
        stdio: "inherit",
        env: environment,
        cwd: paths.session,
      })
      return new Promise<void>((resolveExit, reject) => {
        child.once("exit", () => resolveExit())
        child.once("error", reject)
      })
    })
  }))

  yield* simulationStep("unable to remove the simulation session", () =>
    rm(paths.session, { recursive: true, force: true }))
  yield* Console.log("\nSimulation ended; the simulated world was removed.")
}).pipe(
  Effect.catchTags({
    SimulationError: (error: SimulationError) =>
      Console.error(`Distribution simulation failed: ${error.message}`).pipe(
        Effect.zipRight(Effect.sync(() => {
          process.exitCode = 1
        })),
      ),
    LocalReleaseError: (error) =>
      Console.error(`Distribution simulation failed: ${error.message}`).pipe(
        Effect.zipRight(Effect.sync(() => {
          process.exitCode = 1
        })),
      ),
  }),
)

BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
