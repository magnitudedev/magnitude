import { FileSystem } from "@effect/platform"
import { Effect, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { AssertionFailure, Digest, InfrastructureFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { checkedCommand } from "../process"

const fail = (message: string) => new AssertionFailure({ message })
export const hashFile = (path: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const hash = createHash("sha256")
  yield* fs.stream(path).pipe(Stream.runForEach(bytes => Effect.sync(() => { hash.update(bytes) })))
  return Digest.make(hash.digest("hex"))
})

export const PackagePayload = Schema.Struct({ version: Schema.String, files: Schema.Array(Schema.Struct({ path: Schema.String,
  sha256: Digest, bytes: Schema.Number })) })

/** Extract only for comparison. The package manager and app updater already own installation. */
export const verifyDebPayload = (app: InstalledApplication) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const candidate = app.candidate
  if (candidate.target.packageFormat !== "deb") return yield* new InfrastructureFailure({ operation: "update-payload", message: "This native payload verifier requires a DEB" })
  if ((yield* hashFile(candidate.path)) !== candidate.artifact.sha256 || Number((yield* fs.stat(candidate.path)).size) !== candidate.artifact.bytes) return yield* fail("Admitted package changed before payload verification")
  const extracted = yield* fs.makeTempDirectoryScoped({ prefix: "lab-updated-deb-" })
  yield* checkedCommand("dpkg-deb", ["-x", candidate.path, extracted], { timeoutMs: 120_000 })
  const files: (typeof PackagePayload.Type.files[number])[] = []
  const visit = (relative: string): Effect.Effect<void, unknown, FileSystem.FileSystem> => Effect.gen(function* () {
    const parent = join(extracted, relative)
    for (const name of yield* fs.readDirectory(parent)) {
      const entry = join(relative, name), expected = join(extracted, entry), installed = join("/", entry)
      // lstat distinguishes package symlinks from their live destinations.
      const isLink = yield* fs.readLink(expected).pipe(Effect.option)
      if (isLink._tag === "Some") {
        if ((yield* fs.readLink(installed)) !== isLink.value) return yield* fail(`Installed package symlink differs: ${entry}`)
        continue
      }
      const stat = yield* fs.stat(expected)
      if (stat.type === "Directory") {
        if ((yield* fs.readLink(installed).pipe(Effect.option))._tag === "Some" || (yield* fs.stat(installed)).type !== "Directory") {
          return yield* fail(`Installed directory changed type: ${entry}`)
        }
        yield* visit(entry); continue
      }
      if (stat.type !== "File") return yield* fail(`Unexpected native package entry: ${entry}`)
      if ((yield* fs.readLink(installed).pipe(Effect.option))._tag === "Some") return yield* fail(`Installed regular file became a symlink: ${entry}`)
      const hash = yield* hashFile(expected)
      if ((yield* hashFile(installed)) !== hash || Number((yield* fs.stat(installed)).size) !== Number(stat.size)) return yield* fail(`Installed package payload differs: ${entry}`)
      files.push({ path: installed, sha256: hash, bytes: Number(stat.size) })
    }
  })
  yield* visit("")
  if (!files.length) return yield* fail("DEB contained no payload files")
  return PackagePayload.make({ version: candidate.version, files })
})).pipe(Effect.mapError(error => error instanceof AssertionFailure || error instanceof InfrastructureFailure ? error : new InfrastructureFailure({ operation: "update-payload", message: "Could not compare the installed native package payload" })))
