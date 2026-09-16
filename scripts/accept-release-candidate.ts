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
// Script entry points use Promises; subprocess lifetime and output bounds remain Effect-owned.
const run = (
  command: readonly string[],
  options: {
    readonly cwd?: string
    readonly env?: Readonly<Record<string, string | undefined>>
  } = {},
): Promise<string> => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const executable = command[0]
  if (!executable) return yield* new CandidateAcceptanceFailed({ message: "Empty acceptance command" })
  const environment = Object.fromEntries(Object.entries(options.env ?? {}).filter((entry): entry is [string, string] => entry[1] !== undefined))
  const child = yield* Command.make(executable, ...command.slice(1)).pipe(
    Command.workingDirectory(options.cwd ?? process.cwd()), Command.env(environment), Command.start,
  )
  const read = (stream: typeof child.stdout) => stream.pipe(Stream.decodeText(), Stream.runFoldEffect("", (previous, chunk) =>
    previous.length + chunk.length <= 1024 * 1024 ? Effect.succeed(previous + chunk)
      : Effect.fail(new CandidateAcceptanceFailed({ message: `${executable} exceeded its output limit` }))))
  const [code, stdout, stderr] = yield* Effect.all([child.exitCode, read(child.stdout), read(child.stderr)], { concurrency: "unbounded" })
  if (code !== 0) return yield* new CandidateAcceptanceFailed({ message: `${executable} failed with exit ${code}: ${(stderr || stdout).trim()}` })
  return stdout
})).pipe(Effect.timeout("5 minutes"), Effect.provide(BunContext.layer)))

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
const root = await mkdtemp(resolve(tmpdir(), "magnitude-candidate-"))
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
  const base = yield* selectArtifact(manifest, "icn-base", host)
  const installation = yield* installArtifact(baseUrl, manifest.version, base, resolve(dataDir, "inference"))
  const declaration = resolve(installation, "installation.json")
  yield* fs.writeFileString(declaration, yield* Schema.encode(Schema.parseJson(IcnInstallationDeclaration))({
    schemaVersion: 1, backend: "cpu", nativeBuild: Option.getOrThrow(base.nativeBuild),
    backendModuleAbi: Option.getOrThrow(base.backendModuleAbi),
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

try {
  await Effect.runPromise(acceptBootstrap)
  await invoke([cliExecutable, "--version"], root, resolve(root, "home-cli"))
  server.stop(true)
  await invoke([cliExecutable, "--version"], root, resolve(root, "home-cli"))
  console.log("Desktop-bundled CLI works with the candidate artifact endpoint stopped")
} finally {
  server.stop(true)
  await rm(root, { recursive: true, force: true })
}
