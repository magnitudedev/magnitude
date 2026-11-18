import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option, Schema } from "effect"
import { APPLE_TEAM_ID } from "@magnitudedev/release/trust"
import { nativeWindowsInstallerVerifier, WindowsInstallerVerifier } from "../desktop-native/windows-update-signature"
import { lstat, readdir, readlink, rename, symlink } from "node:fs/promises"
import { posix, win32 } from "node:path"
import { randomBytes } from "node:crypto"
import type { UpdateRelease } from "@magnitudedev/release/hosted-update"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { ApplicationUpdateFailed } from "./application-update"

export class StandaloneInstallationFailed extends Schema.TaggedError<StandaloneInstallationFailed>()("StandaloneInstallationFailed", {
  message: Schema.String,
}) {}

/** Checks that each executable in a staged release carries Magnitude's code signature. */
export interface StandaloneCodeVerifier {
  readonly verify: (path: string) => Effect.Effect<void, StandaloneInstallationFailed>
}
export const StandaloneCodeVerifier = Context.GenericTag<StandaloneCodeVerifier>("@magnitudedev/daemon-management/StandaloneCodeVerifier")
export const unsignedStandaloneCode = StandaloneCodeVerifier.of({ verify: () => Effect.void })

/**
 * The publisher identity compiled into this build: Apple Developer ID team on macOS, Authenticode
 * organization on Windows. Builds without one (unsigned development and branch builds) verify nothing.
 */
export const nativeStandaloneCodeVerifier = (platform: "darwin" | "linux" | "win32", addonPath: string, windowsPublisher: Option.Option<string>) => {
  if (platform === "darwin" && /^[A-Z0-9]{10}$/.test(APPLE_TEAM_ID)) {
    const requirement = `anchor apple generic and certificate leaf[subject.OU] = "${APPLE_TEAM_ID}" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists`
    return Layer.effect(StandaloneCodeVerifier, Effect.map(CommandExecutor.CommandExecutor, executor => StandaloneCodeVerifier.of({ verify: path => Command.make("/usr/bin/codesign",
      "--verify", "--strict", "--all-architectures", "-R", `=${requirement}`, path).pipe(Command.exitCode, Effect.timeout("30 seconds"),
      Effect.filterOrFail(code => code === 0, () => failed(`The code signature of ${path} does not match Magnitude's publisher.`)),
      Effect.asVoid, Effect.mapError(error => error._tag === "StandaloneInstallationFailed" ? error : failed(`Could not check the code signature of ${path}.`)),
      Effect.provideService(CommandExecutor.CommandExecutor, executor)) })))
  }
  if (platform === "win32" && Option.isSome(windowsPublisher)) {
    return Layer.effect(StandaloneCodeVerifier, Effect.map(WindowsInstallerVerifier, verifier => StandaloneCodeVerifier.of({
      verify: path => verifier.verify(path).pipe(Effect.mapError(() => failed(`The code signature of ${path} does not match Magnitude's publisher.`))),
    }))).pipe(Layer.provide(nativeWindowsInstallerVerifier(addonPath, windowsPublisher.value)))
  }
  return Layer.succeed(StandaloneCodeVerifier, unsignedStandaloneCode)
}

type Platform = "darwin" | "linux" | "win32"
const paths = (platform: Platform) => platform === "win32" ? win32 : posix
const executableName = (name: string, platform: Platform) => platform === "win32" ? `${name}.exe` : name

/** The files every standalone release holds; anything else in the archive is refused. */
export const standaloneReleaseFiles = (platform: Platform) => {
  const required = [executableName("magnitude", platform), executableName("magnitude-service", platform), "desktop-host.node",
    "update-configuration.json", "update-trust.json"]
  const optional = ["LICENSE", ...(platform === "win32" ? ["magnitude-launcher.exe"] : ["magnitude-command"])]
  const signed = [executableName("magnitude", platform), executableName("magnitude-service", platform), "desktop-host.node",
    ...(platform === "win32" ? ["magnitude-launcher.exe"] : platform === "darwin" ? ["magnitude-command"] : [])]
  return { required, allowed: new Set([...required, ...optional]), signed }
}

const failed = (message: string) => new StandaloneInstallationFailed({ message })
const io = <A>(message: string, run: () => Promise<A>) => Effect.tryPromise({ try: run, catch: error => failed(`${message}: ${String(error)}`) })

const tarExecutable = (platform: Platform, environment: Readonly<Record<string, string | undefined>>) => platform === "win32"
  ? win32.join(environment.SystemRoot ?? "C:\\Windows", "System32", "tar.exe")
  : platform === "darwin" ? "/usr/bin/tar" : "tar"

/** Which version `current` names, or none before the first installation. */
export const readStandaloneCurrent = (root: string, platform: Platform) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const pointer = paths(platform).join(root, "current")
  if (platform === "win32") return yield* fs.readFileString(pointer).pipe(Effect.map(text => Option.some(text.trim())), Effect.orElseSucceed(() => Option.none<string>()))
  return yield* io("Could not read the current version", () => readlink(pointer)).pipe(Effect.map(target => Option.some(posix.basename(target))),
    Effect.orElseSucceed(() => Option.none<string>()))
})

/** Atomically names `version` as current: a renamed symlink on macOS and Linux, a renamed pointer file on Windows. */
export const switchStandaloneCurrent = (root: string, version: string, platform: Platform) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = paths(platform)
  const temporary = path.join(root, `.current-${randomBytes(6).toString("hex")}`)
  if (platform === "win32") yield* fs.writeFileString(temporary, `${version}\n`).pipe(Effect.mapError(error => failed(`Could not write the current version: ${error.message}`)))
  else yield* io("Could not link the current version", () => symlink(version, temporary))
  yield* io("Could not switch the current version", () => rename(temporary, path.join(root, "current")))
})

/**
 * Unpacks a verified release archive beside the running version, checks its files, signatures and
 * version, then makes it current. Versions other than the new one and `keep` are removed afterwards.
 * Returns the new version's `magnitude` executable.
 */
export const installStandaloneArchive = (options: {
  readonly root: string
  readonly archive: string
  readonly version: string
  readonly keep: string
  readonly platform: Platform
  readonly environment: Readonly<Record<string, string | undefined>>
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const verifier = yield* StandaloneCodeVerifier
  const path = paths(options.platform)
  const files = standaloneReleaseFiles(options.platform)
  if (!/^[0-9A-Za-z.+-]+$/.test(options.version) || options.version.startsWith(".")) return yield* failed("The release version is not a valid folder name.")
  const staging = path.join(options.root, `.staging-${options.version}`)
  const target = path.join(options.root, options.version)
  yield* fs.remove(staging, { recursive: true }).pipe(Effect.ignore)
  yield* fs.makeDirectory(staging, { recursive: true, mode: 0o755 }).pipe(Effect.mapError(error => failed(`Could not prepare the update folder: ${error.message}`)))
  const unpacked = yield* Command.make(tarExecutable(options.platform, options.environment), "-xzf", options.archive, "-C", staging, "--no-same-owner").pipe(
    Command.exitCode, Effect.timeout("5 minutes"), Effect.mapError(error => failed(`Could not unpack the update: ${String(error)}`)))
  if (unpacked !== 0) return yield* failed("Could not unpack the update.")
  for (const name of yield* io("Could not read the unpacked update", () => readdir(staging))) {
    const info = yield* io("Could not inspect the unpacked update", () => lstat(path.join(staging, name)))
    if (!files.allowed.has(name) || !info.isFile()) return yield* failed(`The update contains an unexpected entry: ${name}`)
  }
  for (const name of files.required) {
    if (!(yield* fs.exists(path.join(staging, name)).pipe(Effect.orElseSucceed(() => false)))) return yield* failed(`The update is missing ${name}.`)
  }
  for (const name of files.signed) {
    const file = path.join(staging, name)
    if (yield* fs.exists(file).pipe(Effect.orElseSucceed(() => false))) yield* verifier.verify(file)
  }
  const reported = yield* Command.make(path.join(staging, executableName("magnitude", options.platform)), "--version").pipe(
    Command.string, Effect.timeout("30 seconds"), Effect.map(text => text.trim()), Effect.mapError(() => failed("The updated command did not start.")))
  if (reported !== options.version) return yield* failed(`The updated command reports version ${reported}, not ${options.version}.`)
  if (options.version !== options.keep) yield* fs.remove(target, { recursive: true }).pipe(Effect.ignore)
  yield* io("Could not move the update into place", () => rename(staging, target))
  yield* switchStandaloneCurrent(options.root, options.version, options.platform)
  yield* pruneStandaloneVersions(options.root, [options.version, options.keep], options.platform)
  return path.join(target, executableName("magnitude", options.platform))
})

/** Removes version folders and leftovers other than `keep`. A folder still in use stays for the next attempt. */
export const pruneStandaloneVersions = (root: string, keep: ReadonlyArray<string>, platform: Platform) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const entries = yield* io("Could not list installed versions", () => readdir(root)).pipe(Effect.orElseSucceed(() => [] as string[]))
  for (const name of entries) {
    if (name === "current" || keep.includes(name)) continue
    yield* fs.remove(paths(platform).join(root, name), { recursive: true }).pipe(Effect.ignore)
  }
})

/**
 * Installs the prepared release into a standalone installation. The caller holds maintenance and
 * installation admission with no running service; on success it restarts into the returned executable.
 */
export const completeStandaloneUpdate = (options: {
  readonly root: string
  readonly runningVersion: string
  readonly platform: Platform
  readonly environment: Readonly<Record<string, string | undefined>>
}) => Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  const pending = yield* store.read
  if (Option.isNone(pending)) return yield* new ApplicationUpdateFailed({ message: "There is no prepared application update to install." })
  const release: UpdateRelease = pending.value.release
  const archive = yield* store.verify(release).pipe(Effect.tapError(error => store.recordFailure(release, "verify", error.message)))
  yield* store.recordAttempt(release)
  const executable = yield* installStandaloneArchive({ root: options.root, archive, version: release.version, keep: options.runningVersion,
    platform: options.platform, environment: options.environment }).pipe(
    Effect.tapError(error => store.recordFailure(release, "install", error.message)))
  yield* store.complete(release)
  return { version: release.version, executable }
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
