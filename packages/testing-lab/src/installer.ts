import { isWindows } from "./domain"
import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option, Schema } from "effect"
import { join, resolve } from "node:path"
import { createHash } from "node:crypto"
import { Stream } from "effect"
import { Candidate } from "./candidate"
import { AssertionFailure, InfrastructureFailure } from "./domain"
import { command, ProcessExecutor } from "./process"

export const InstalledApplication = Schema.Struct({ candidate: Candidate, root: Schema.String, executable: Schema.String, cli: Schema.String,
  packageVersion: Schema.String })
export type InstalledApplication = typeof InstalledApplication.Type
export interface Installer {
  readonly install: (candidate: Candidate) => Effect.Effect<InstalledApplication, AssertionFailure | InfrastructureFailure>
  readonly uninstall: (application: InstalledApplication) => Effect.Effect<void, AssertionFailure | InfrastructureFailure>
}
export const Installer = Context.GenericTag<Installer>("@magnitudedev/testing-lab/Installer")
export const InstallerConfig = Schema.Struct({
  /** System package managers and Windows registration require a disposable worker/user. */
  disposable: Schema.Boolean, root: Schema.String, environment: Schema.Record({ key: Schema.String, value: Schema.String }),
})
const fail = (message: string) => new AssertionFailure({ message })
const infra = (message: string) => new InfrastructureFailure({ operation: "installer", message })
const errors = <A, E extends { readonly message: string; readonly _tag: string }, R>(effect: Effect.Effect<A, E, R>) => effect.pipe(
  Effect.mapError(error => error._tag === "AssertionFailure" ? fail(error.message) : infra(error.message)))

export const nativeInstaller = (config: typeof InstallerConfig.Type) => Layer.effect(Installer, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* ProcessExecutor
  const run = (executable: string, args: readonly string[], timeoutMs = 300_000, environment = config.environment) => command(executable, args, { env: environment,
    inheritEnv: false, timeoutMs }).pipe(Effect.provideService(ProcessExecutor, executor))
  const checked = (executable: string, args: readonly string[]) => run(executable, args).pipe(Effect.flatMap(result => result.exitCode === 0 ? Effect.succeed(result.stdout)
    : Effect.fail(fail(`${executable} exited ${result.exitCode}:\nstdout: ${result.stdout.slice(-2400)}\nstderr: ${result.stderr.slice(-2400)}`))))
  // Start-Process -Wait includes the NSIS uninstaller's copied child process.
  const windowsInstaller = (path: string) => run("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command",
    '$ErrorActionPreference = "Stop"; $p = Start-Process -FilePath $env:LAB_NATIVE_INSTALLER -ArgumentList "/S" -PassThru -Wait; exit $p.ExitCode'],
    300_000, { ...config.environment, LAB_NATIVE_INSTALLER: path }).pipe(Effect.flatMap(result => result.exitCode === 0 ? Effect.void
      : Effect.fail(fail(`Windows installer exited ${result.exitCode}: ${(result.stderr || result.stdout).slice(-1800)}`))))
  const verifyBytes = (candidate: Candidate) => Effect.gen(function* () {
    const hash = createHash("sha256")
    let size = 0
    yield* fs.stream(candidate.path).pipe(Stream.runForEach(bytes => Effect.sync(() => { hash.update(bytes); size += bytes.byteLength })))
    if (size !== candidate.artifact.bytes || hash.digest("hex") !== candidate.artifact.sha256) return yield* fail("Installer changed after download; refusing installation")
  })
  const host = (candidate: Candidate) => Effect.gen(function* () {
    if (candidate.artifact.kind !== "desktop" || !Option.contains(candidate.artifact.host, candidate.target.artifactHost)
      || !candidate.artifact.filename.endsWith(`.${candidate.target.packageFormat}`)) return yield* fail("Installer metadata differs from requested target")
    const platform = candidate.target.os === "macos" ? "darwin" : isWindows(candidate.target.os) ? "win32" : "linux"
    if (process.platform !== platform || process.arch !== candidate.target.arch) return yield* infra("Native installer host differs from requested target")
    if (platform !== "darwin" && !config.disposable) return yield* infra("System installation requires a disposable lab worker/user")
  })
  const packageVersion = (candidate: Candidate) => candidate.target.packageFormat === "deb"
    ? checked("dpkg-query", ["-W", "-f=${Version}", "magnitude-desktop"]).pipe(Effect.map(s => s.trim()))
    : checked("rpm", ["-q", "--qf", "%{VERSION}-%{RELEASE}", "magnitude-desktop"]).pipe(Effect.map(s => s.trim()))
  const install = (candidate: Candidate) => errors(Effect.gen(function* () {
    yield* host(candidate)
    yield* verifyBytes(candidate)
    const format = candidate.target.packageFormat
    let root: string, executable: string, cli: string, version = candidate.version
    if (format === "dmg") {
      yield* fs.makeDirectory(config.root, { recursive: true, mode: 0o700 })
      root = join(yield* fs.realPath(config.root), "Magnitude.app")
      if (yield* fs.exists(root)) return yield* fail("An application already exists at the lab installation path")
      yield* Effect.scoped(Effect.gen(function* () {
        const mount = yield* fs.makeTempDirectoryScoped({ prefix: "lab-dmg-" })
        yield* Effect.acquireRelease(checked("/usr/bin/hdiutil", ["attach", "-readonly", "-nobrowse", "-mountpoint", mount, candidate.path]),
          () => checked("/usr/bin/hdiutil", ["detach", mount]).pipe(Effect.orDie))
        if (!(yield* fs.exists(join(mount, "Magnitude.app", "Contents", "MacOS", "Magnitude")))) return yield* fail("Installer contains no Magnitude application")
        yield* checked("/usr/bin/ditto", [join(mount, "Magnitude.app"), root])
      }))
      executable = join(root, "Contents", "MacOS", "Magnitude")
      cli = join(root, "Contents", "Resources", "magnitude")
      version = (yield* checked("/usr/bin/plutil", ["-extract", "CFBundleShortVersionString", "raw", "-o", "-", join(root, "Contents", "Info.plist")])).trim()
      if (version !== candidate.version) return yield* fail("Installed application version differs from the candidate manifest")
    } else if (format === "exe") {
      const localAppData = config.environment.LOCALAPPDATA
      if (!localAppData) return yield* infra("Disposable Windows user's LOCALAPPDATA is required")
      root = join(localAppData, "Programs", "Magnitude")
      if (yield* fs.exists(root)) return yield* fail("Disposable user already has a Magnitude installation")
      yield* windowsInstaller(candidate.path)
      executable = join(root, "Magnitude.exe")
      cli = join(root, "resources", "magnitude.exe")
    } else {
      const present = yield* run(format === "deb" ? "dpkg-query" : "rpm", format === "deb" ? ["-W", "-f=${db:Status-Status}", "magnitude-desktop"] : ["-q", "magnitude-desktop"])
      if (present.exitCode === 0 && (format !== "deb" || present.stdout.trim() === "installed")) return yield* fail("Disposable worker already has a Magnitude installation")
      yield* checked("sudo", ["-n", format === "deb" ? "apt-get" : "dnf", "install", "-y", candidate.path])
      root = "/usr/lib/magnitude-desktop"
      executable = "/usr/bin/magnitude-desktop"
      cli = "/usr/bin/magnitude"
      version = yield* packageVersion(candidate)
    }
    if (!(yield* fs.exists(executable)) || !(yield* fs.exists(cli))) return yield* fail("Native installer did not install both application and bundled CLI")
    return InstalledApplication.make({ candidate, root, executable, cli, packageVersion: version })
  }))
  const uninstall = (application: InstalledApplication) => errors(Effect.gen(function* () {
    yield* host(application.candidate)
    const format = application.candidate.target.packageFormat
    if (format === "dmg") {
      const expected = join(yield* fs.realPath(config.root), "Magnitude.app")
      if (resolve(application.root) !== expected || (yield* fs.realPath(application.root)) !== expected) return yield* infra("Application removal path differs from owned installation")
      yield* fs.remove(application.root, { recursive: true })
    } else if (format === "exe") {
      const expected = join(config.environment.LOCALAPPDATA ?? "", "Programs", "Magnitude")
      if (application.root !== expected) return yield* infra("Windows installation identity changed")
      yield* windowsInstaller(join(application.root, "Uninstall Magnitude.exe"))
    } else {
      if ((yield* packageVersion(application.candidate)) !== application.packageVersion) return yield* infra("Installed package changed; refusing to uninstall a different version")
      yield* checked("sudo", ["-n", format === "deb" ? "apt-get" : "dnf", "remove", "-y", "magnitude-desktop"])
    }
    if (yield* fs.exists(application.executable)) return yield* fail("Application executable remains after native removal")
  }))
  return { install, uninstall } satisfies Installer
}))
