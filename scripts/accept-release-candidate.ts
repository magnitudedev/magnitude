import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema, Stream } from "effect"
import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { IcnInstallationDeclaration } from "@magnitudedev/icn-protocol"
import { NodeArchiveExtractor } from "../packages/release/src/archive"
import { currentHost } from "@magnitudedev/release/targets"
import { sha256File } from "@magnitudedev/release/macos-app"
import { validateDesktopDistribution } from "../packages/release/scripts/apple/desktop"
import { validateLinuxDesktopInstaller } from "../packages/release/scripts/build/desktop-linux"
import {
  mkdtemp,
  readFile,
  rm,
} from "node:fs/promises"
import { tmpdir } from "node:os"
import { resolve } from "node:path"
import { releaseUrl, installArtifact, selectArtifact } from "@magnitudedev/release/acquisition"
import { ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"

class CandidateAcceptanceFailed extends Schema.TaggedError<CandidateAcceptanceFailed>()("CandidateAcceptanceFailed", { message: Schema.String }) {}

const candidate = resolve(process.argv[2] ?? "release-candidate")
// Script entry points use Promises; subprocess lifetime remains Effect-owned.
const run = (
  command: readonly string[],
  options: {
    readonly cwd?: string
    readonly env?: Readonly<Record<string, string | undefined>>
    readonly timeout?: "5 minutes" | "25 minutes"
  } = {},
): Promise<string> => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const executable = command[0]
  if (!executable) return yield* new CandidateAcceptanceFailed({ message: "Empty acceptance command" })
  const environment = Object.fromEntries(Object.entries(options.env ?? {}).filter((entry): entry is [string, string] => entry[1] !== undefined))
  const child = yield* Command.make(executable, ...command.slice(1)).pipe(
    Command.workingDirectory(options.cwd ?? process.cwd()), Command.env(environment), Command.start,
  )
  const read = (stream: typeof child.stdout) => stream.pipe(Stream.decodeText(), Stream.runFold("", (previous, chunk) => previous + chunk))
  const [code, stdout, stderr] = yield* Effect.all([child.exitCode, read(child.stdout), read(child.stderr)], { concurrency: "unbounded" })
  // Report both streams: a process may log to stderr while its failure is on stdout.
  if (code !== 0) return yield* new CandidateAcceptanceFailed({
    message: `${executable} failed with exit ${code}\n--- stdout ---\n${stdout.trim()}\n--- stderr ---\n${stderr.trim()}`,
  })
  return stdout
})).pipe(Effect.timeout(options.timeout ?? "5 minutes"), Effect.provide(BunContext.layer)))

const manifest = await Effect.runPromise(Schema.decodeUnknown(Schema.parseJson(ReleaseManifestSchema))(
  await readFile(resolve(candidate, "magnitude-release.json"), "utf8"),
).pipe(Effect.flatMap(validateReleaseManifest)))

const routes = new Map(
  [
    "magnitude-release.json",
    ...manifest.artifacts.map((artifact) => artifact.filename),
  ].map((name) => [
    new URL(releaseUrl("http://release.invalid", manifest.version, name)).pathname,
    name,
  ]),
)
const server = Bun.serve({
  port: 0,
  hostname: "127.0.0.1",
  async fetch(request) {
    const name = routes.get(new URL(request.url).pathname)
    if (!name) return new Response("missing", { status: 404 })
    try {
      return new Response(await readFile(resolve(candidate, name)))
    } catch {
      return new Response("missing", { status: 404 })
    }
  },
})
const baseUrl = `http://127.0.0.1:${server.port}`
const diagnosticParent = process.env.MAGNITUDE_CANDIDATE_DIAGNOSTICS ?? tmpdir()
const root = await mkdtemp(resolve(diagnosticParent, "magnitude-candidate-"))
const headlessRoot = await mkdtemp(resolve(diagnosticParent, "mag-candidate-headless-"))
const dataDir = resolve(root, "home-bootstrap", ".magnitude")
let desktopApplication = "/usr/bin/magnitude-desktop"
let cliExecutable = "/usr/bin/magnitude"
const environment = (home: string) => ({
  ...process.env,
  HOME: home,
  USERPROFILE: home,
  MAGNITUDE_DESKTOP_PATH: desktopApplication,
  MAGNITUDE_RELEASE_BASE_URL: baseUrl,
})

/** Candidate acceptance launches the sealed desktop; it never owns a standalone daemon. */
const acceptBootstrap = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const host = currentHost()
  const linux = host === "linux-arm64-gnu" || host === "linux-x64-gnu"
  if (!linux && host !== "darwin-arm64" && host !== "darwin-x64") return yield* new CandidateAcceptanceFailed({
    message: "Candidate desktop installer acceptance is not implemented for this host yet",
  })
  const desktops = manifest.artifacts.filter(value => value.kind === "desktop" && Option.getOrUndefined(value.host) === host &&
    (value.id === (linux ? `desktop-${host}-deb` : `desktop-${host}`)))
  if (desktops.length !== 1) return yield* new CandidateAcceptanceFailed({ message: "Candidate must contain exactly one selected desktop installer" })
  const desktop = desktops[0]!
  const image = resolve(candidate, desktop.filename)
  const info = yield* fs.stat(image)
  if (Number(info.size) !== desktop.bytes || (yield* sha256File(image)) !== desktop.sha256) {
    return yield* new CandidateAcceptanceFailed({ message: "Desktop installer differs from the candidate manifest" })
  }
  const inference = yield* selectArtifact(manifest, "icn-base", host)
  const installation = yield* installArtifact(baseUrl, manifest.version, inference, resolve(dataDir, "inference"))
  const declaration = resolve(installation, "installation.json")
  yield* fs.writeFileString(declaration, yield* Schema.encode(Schema.parseJson(IcnInstallationDeclaration))({
    schemaVersion: 1, nativeBuild: Option.getOrThrow(inference.nativeBuild),
  }))
  if (linux) {
    yield* validateLinuxDesktopInstaller({ file: image, format: "deb", arch: host === "linux-arm64-gnu" ? "arm64" : "x64",
      version: manifest.version, revision: manifest.acnRevision })
    // Linux candidate acceptance runs on a disposable consumer with native package installation.
    const installed = yield* Command.make("sudo", "apt-get", "install", "-y", "--reinstall", "--no-install-recommends", image).pipe(
      Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
    )
    if (installed !== 0) return yield* new CandidateAcceptanceFailed({ message: "Candidate DEB installation failed" })
    const code = yield* Command.make("xvfb-run", "-a", "dbus-run-session", "--", process.env.MAGNITUDE_TEST_NODE ?? "node",
      resolve(import.meta.dir, "../desktop/src/fixtures/linux-installed-lifecycle.mjs")).pipe(
      Command.env({ MAGNITUDE_TEST_CLI_EXECUTABLE: cliExecutable, MAGNITUDE_TEST_INFERENCE_INSTALLATION: declaration }),
      Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode, Effect.timeout("3 minutes"),
    )
    if (code !== 0) return yield* new CandidateAcceptanceFailed({ message: "Candidate Linux desktop lifecycle failed" })
  } else {
    const updates = manifest.artifacts.filter(value => value.kind === "desktop" && Option.getOrUndefined(value.host) === host && value.id === `desktop-update-${host}`)
    if (updates.length !== 1) return yield* new CandidateAcceptanceFailed({ message: "Candidate must contain exactly one Mac update archive" })
    const update = updates[0]!
    const updateArchive = resolve(candidate, update.filename)
    if (Number((yield* fs.stat(updateArchive)).size) !== update.bytes || (yield* sha256File(updateArchive)) !== update.sha256) {
      return yield* new CandidateAcceptanceFailed({ message: "Desktop update archive differs from the candidate manifest" })
    }
    yield* validateDesktopDistribution({ image, updateArchive, version: manifest.version, revision: manifest.acnRevision,
      rpcVersion: manifest.rpc.version, inferenceInstallation: declaration })
    const desktopRoot = resolve(root, "desktop")
    yield* Command.make("/usr/bin/ditto", "-x", "-k", updateArchive, desktopRoot).pipe(Command.string)
    desktopApplication = resolve(desktopRoot, "Magnitude.app")
    cliExecutable = resolve(desktopApplication, "Contents/Resources/magnitude")
  }
})).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer, NodeArchiveExtractor]))

const invoke = async (
  command: readonly string[],
  directory: string,
  home: string,
): Promise<void> => {
  const output = (await run(command, {
    cwd: directory,
    env: environment(home),
  })).trim()
  if (output !== manifest.version) {
    throw new Error(`${command[0]} returned ${output}; expected ${manifest.version}`)
  }
}

let accepted = false
try {
  await Effect.runPromise(acceptBootstrap)
  const headlessEnvironment = {
    ...environment(resolve(headlessRoot, "home")),
    MAGNITUDE_INSTALLED_ACCEPTANCE_OUTPUT: headlessRoot,
    MAGNITUDE_INSTALLED_ACCEPTANCE_CLI: cliExecutable,
    MAGNITUDE_INSTALLED_ACCEPTANCE_ADDON: process.platform === "darwin"
      ? resolve(desktopApplication, "Contents/Resources/desktop-host.node")
      : "/usr/lib/magnitude-desktop/resources/desktop-host.node",
    MAGNITUDE_INSTALLED_ACCEPTANCE_VERSION: manifest.version,
    MAGNITUDE_ICN_PATH: "",
  }
  await run([process.execPath, resolve(import.meta.dir, "../packages/release/scripts/acceptance/test-installed-headless.ts")], {
    cwd: root,
    env: headlessEnvironment,
    timeout: "25 minutes",
  })
  await Effect.runPromise(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ engineAcquired: Schema.Literal(true), rankingReady: Schema.Literal(true) })))(
    await readFile(resolve(headlessRoot, "result.json"), "utf8"),
  ))
  console.log("Installed desktop acquired ICN into an empty profile and reached headless readiness")
  await invoke([cliExecutable, "--version"], root, resolve(root, "home-cli"))
  server.stop(true)
  await run([process.execPath, resolve(import.meta.dir, "../packages/release/scripts/acceptance/test-installed-headless.ts")], {
    cwd: root,
    env: { ...headlessEnvironment, MAGNITUDE_RELEASE_BASE_URL: "http://127.0.0.1:1", MAGNITUDE_INSTALLED_ACCEPTANCE_OFFLINE: "true" },
    timeout: "25 minutes",
  })
  await Effect.runPromise(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ offlineCachedStart: Schema.Literal(true), rankingReady: Schema.Literal(true) })))(
    await readFile(resolve(headlessRoot, "result.json"), "utf8"),
  ))
  await invoke([cliExecutable, "--version"], root, resolve(root, "home-cli"))
  console.log("Installed service and bundled CLI work with the candidate artifact endpoint stopped")
  accepted = true
} finally {
  server.stop(true)
  if (accepted) {
    await rm(headlessRoot, { recursive: true, force: true })
    await rm(root, { recursive: true, force: true })
  } else {
    console.error(`Candidate diagnostics preserved at ${headlessRoot} and ${root}`)
  }
}
