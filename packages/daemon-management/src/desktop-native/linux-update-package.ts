import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { Context, Effect, Option, Schema, Stream } from "effect"
import { createHash, type KeyObject } from "node:crypto"
import { join } from "node:path"
import { LINUX_DESKTOP_PACKAGE_NAME } from "@magnitudedev/release/executables"
import { acceptsUpdateManifest, SignedUpdateManifest, verifyUpdateManifest } from "@magnitudedev/release/hosted-update"

export const LinuxPackageUpdate = Schema.Struct({
  envelope: SignedUpdateManifest,
  packagePath: Schema.NonEmptyString,
})
export class LinuxPackageUpdateFailed extends Schema.TaggedError<LinuxPackageUpdateFailed>()("LinuxPackageUpdateFailed", {
  message: Schema.String,
}) {}
export interface LinuxPackageInstaller {
  readonly install: (request: typeof LinuxPackageUpdate.Type) => Effect.Effect<void, LinuxPackageUpdateFailed>
}
export const LinuxPackageInstaller = Context.GenericTag<LinuxPackageInstaller>("@magnitudedev/daemon-management/LinuxPackageInstaller")

/** The privileged CLI composition root supplies installed, root-owned publisher trust. */
export const makeLinuxPackageInstaller = (options: {
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
  readonly currentVersion: string
  readonly package: "deb" | "rpm"
  readonly callerUid: number
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* CommandExecutor.CommandExecutor
  return LinuxPackageInstaller.of({ install: request => Effect.scoped(Effect.gen(function* () {
    if (process.platform !== "linux" || process.getuid?.() !== 0 || !Number.isSafeInteger(options.callerUid) || options.callerUid <= 0) {
      return yield* new LinuxPackageUpdateFailed({ message: "Package installation requires system authorization from your desktop session." })
    }
    const manifest = yield* verifyUpdateManifest(request.envelope, options.trustedPublishers)
    const target = manifest.artifact.target
    if (target.os !== "linux" || (process.arch !== "arm64" && process.arch !== "x64")
      || !acceptsUpdateManifest(manifest, { version: options.currentVersion, os: "linux", arch: process.arch, package: options.package })) {
      return yield* new LinuxPackageUpdateFailed({ message: "This package is not a newer Magnitude release for this machine." })
    }
    const source = yield* fs.stat(request.packagePath)
    if (source.type !== "File" || Number(source.size) !== manifest.artifact.bytes || Option.getOrUndefined(source.uid) !== options.callerUid) {
      return yield* new LinuxPackageUpdateFailed({ message: "The downloaded update is missing or has changed." })
    }
    // Copy before verification so an unprivileged writer cannot change the bytes installed as root.
    const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-package-update-" })
    yield* fs.chmod(directory, 0o700)
    const archive = join(directory, `magnitude.${target.package}`)
    let copied = 0
    const digest = createHash("sha256")
    yield* fs.stream(request.packagePath).pipe(Stream.tap(bytes => Effect.gen(function* () {
      copied += bytes.length
      if (copied > manifest.artifact.bytes) return yield* new LinuxPackageUpdateFailed({ message: "The downloaded update grew during preparation." })
      digest.update(bytes)
    })), Stream.run(fs.sink(archive, { flag: "wx", mode: 0o600 })))
    yield* fs.chmod(archive, 0o400)
    if (Number((yield* fs.stat(archive)).size) !== manifest.artifact.bytes) {
      return yield* new LinuxPackageUpdateFailed({ message: "The downloaded update size changed during preparation." })
    }
    if (digest.digest("hex") !== manifest.artifact.sha256) {
      return yield* new LinuxPackageUpdateFailed({ message: "The downloaded update failed publisher verification." })
    }
    const query = target.package === "deb"
      ? Command.make("/usr/bin/dpkg-deb", "--show", "--showformat=${Package}\t${Version}\t${Architecture}", archive)
      : Command.make("/usr/bin/rpm", "-qp", "--qf", "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}", archive)
    const identity = (yield* executor.string(query)).trim().split("\t")
    const arch = target.package === "deb" ? target.arch === "arm64" ? "arm64" : "amd64" : target.arch === "arm64" ? "aarch64" : "x86_64"
    const versionPrefix = `${manifest.version.replace("-", "~")}-`
    if (identity.length !== 3 || identity[0] !== LINUX_DESKTOP_PACKAGE_NAME || identity[2] !== arch
      || !identity[1]!.startsWith(versionPrefix) || !/^[1-9][0-9]*$/.test(identity[1]!.slice(versionPrefix.length))) {
      return yield* new LinuxPackageUpdateFailed({ message: "The package identity does not match the signed Magnitude release." })
    }
    const install = target.package === "deb"
      ? Command.make("/usr/bin/apt-get", "install", "--yes", "--no-remove", "--", archive).pipe(Command.env({ DEBIAN_FRONTEND: "noninteractive", NEEDRESTART_MODE: "l" }))
      : Command.make("/usr/bin/dnf", "--assumeyes", "install", archive)
    const status = yield* executor.exitCode(install.pipe(Command.stdout("inherit"), Command.stderr("inherit")))
    if (status !== 0) return yield* new LinuxPackageUpdateFailed({ message: "The system package manager could not install Magnitude. Check its installation details before retrying." })
  })).pipe(Effect.mapError(error => error instanceof LinuxPackageUpdateFailed ? error
    : new LinuxPackageUpdateFailed({ message: "The signed application package could not be verified or installed." }))) })
})
