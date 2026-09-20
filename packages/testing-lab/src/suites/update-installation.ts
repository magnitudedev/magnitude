import { FileSystem } from "@effect/platform"
import { Effect, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { Candidate } from "../candidate"
import { AssertionFailure, Digest, InfrastructureFailure, Target } from "../domain"
import { InstalledApplication } from "../installer"
import { checkedCommand } from "../process"

const fail = (message: string) => new AssertionFailure({ message })
const hashFile = (path: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const hash = createHash("sha256")
  yield* fs.stream(path).pipe(Stream.runForEach(bytes => Effect.sync(() => { hash.update(bytes) })))
  return Digest.make(hash.digest("hex"))
})

/** Read the native package database; never install the proposed replacement to satisfy an update. */
export const observeUpdatedInstallation = (previous: InstalledApplication, candidate: Candidate, environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  if (!Schema.equivalence(Target)(previous.candidate.target, candidate.target)) return yield* fail("Updated installation target changed")
  const run = (name: string, args: readonly string[]) => checkedCommand(name, args, { env: { ...environment }, inheritEnv: false, timeoutMs: 30_000 })
  const format = candidate.target.packageFormat
  let version = candidate.version
  if (format === "deb") {
    const expected = (yield* run("dpkg-deb", ["-f", candidate.path, "Version"])).stdout.trim()
    version = (yield* run("dpkg-query", ["-W", "-f=${Version}", "magnitude-desktop"])).stdout.trim()
    if (version !== expected) return yield* fail("Native package database still contains a different DEB version")
  } else if (format === "rpm") {
    const expected = (yield* run("rpm", ["-qp", "--qf", "%{VERSION}-%{RELEASE}", candidate.path])).stdout.trim()
    version = (yield* run("rpm", ["-q", "--qf", "%{VERSION}-%{RELEASE}", "magnitude-desktop"])).stdout.trim()
    if (version !== expected) return yield* fail("Native package database still contains a different RPM version")
  } else if (format === "dmg") {
    version = (yield* run("/usr/bin/plutil", ["-extract", "CFBundleShortVersionString", "raw", "-o", "-", join(previous.root, "Contents/Info.plist")])).stdout.trim()
    if (version !== candidate.version) return yield* fail("Updated bundle version differs from its candidate")
  }
  const cli = yield* run(previous.cli, ["--version"])
  if (cli.stdout.trim() !== candidate.version) return yield* fail("Updated bundled CLI version differs from its candidate")
  return InstalledApplication.make({ ...previous, candidate, packageVersion: version })
})

export const UpdatedPayload = Schema.Struct({ version: Schema.String, files: Schema.Array(Schema.Struct({ path: Schema.String,
  sha256: Digest, bytes: Schema.Number })) })

/** Extract only for comparison. The package manager and app updater already own installation. */
export const verifyUpdatedDebPayload = (app: InstalledApplication) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const candidate = app.candidate
  if (candidate.target.packageFormat !== "deb") return yield* new InfrastructureFailure({ operation: "update-payload", message: "This native payload verifier requires a DEB" })
  if ((yield* hashFile(candidate.path)) !== candidate.artifact.sha256 || Number((yield* fs.stat(candidate.path)).size) !== candidate.artifact.bytes) return yield* fail("Admitted update package changed before payload verification")
  const extracted = yield* fs.makeTempDirectoryScoped({ prefix: "lab-updated-deb-" })
  yield* checkedCommand("dpkg-deb", ["-x", candidate.path, extracted], { timeoutMs: 120_000 })
  const files: (typeof UpdatedPayload.Type.files[number])[] = []
  const visit = (relative: string): Effect.Effect<void, unknown, FileSystem.FileSystem> => Effect.gen(function* () {
    const parent = join(extracted, relative)
    for (const name of yield* fs.readDirectory(parent)) {
      const entry = join(relative, name), expected = join(extracted, entry), installed = join("/", entry)
      // lstat distinguishes package symlinks from their live destinations.
      const isLink = yield* fs.readLink(expected).pipe(Effect.option)
      if (isLink._tag === "Some") {
        if ((yield* fs.readLink(installed)) !== isLink.value) return yield* fail(`Updated package symlink differs: ${entry}`)
        continue
      }
      const stat = yield* fs.stat(expected)
      if (stat.type === "Directory") { yield* visit(entry); continue }
      if (stat.type !== "File") return yield* fail(`Unexpected native package entry: ${entry}`)
      if ((yield* fs.readLink(installed).pipe(Effect.option))._tag === "Some") return yield* fail(`Updated regular file became a symlink: ${entry}`)
      const hash = yield* hashFile(expected)
      if ((yield* hashFile(installed)) !== hash || Number((yield* fs.stat(installed)).size) !== Number(stat.size)) return yield* fail(`Updated package payload differs: ${entry}`)
      files.push({ path: installed, sha256: hash, bytes: Number(stat.size) })
    }
  })
  yield* visit("")
  if (!files.length) return yield* fail("Updated DEB contained no payload files")
  return UpdatedPayload.make({ version: candidate.version, files })
})).pipe(Effect.mapError(error => error instanceof AssertionFailure || error instanceof InfrastructureFailure ? error : new InfrastructureFailure({ operation: "update-payload", message: "Could not compare the updated native package payload" })))
