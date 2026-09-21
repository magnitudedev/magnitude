import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure, Digest, InfrastructureFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { hashFile, PackagePayload } from "./package-payload"

const Entry = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("file"), path: Schema.String, sha256: Digest, bytes: Schema.Number }),
  Schema.Struct({ kind: Schema.Literal("directory"), path: Schema.String }),
  Schema.Struct({ kind: Schema.Literal("link"), path: Schema.String, destination: Schema.String }),
)
export const InstalledPayload = Schema.Struct({ version: Schema.String, installer: Digest, entries: Schema.Array(Entry) })

/** Capture before launch, from a clean native installation of the admitted candidate. */
export const installedPayload = (app: InstalledApplication) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if ((yield* fs.readLink(app.root).pipe(Effect.option))._tag === "Some") {
    return yield* new AssertionFailure({ message: "Installed payload root became a symlink" })
  }
  const entries: (typeof Entry.Type)[] = []
  const visit = (path: string): Effect.Effect<void, unknown, FileSystem.FileSystem> => Effect.gen(function* () {
    const absolute = join(app.root, path)
    const link = yield* fs.readLink(absolute).pipe(Effect.option)
    if (link._tag === "Some") { entries.push({ kind: "link", path, destination: link.value }); return }
    const stat = yield* fs.stat(absolute)
    if (stat.type === "Directory") {
      entries.push({ kind: "directory", path })
      for (const name of (yield* fs.readDirectory(absolute)).sort()) yield* visit(path ? `${path}/${name}` : name)
    } else if (stat.type === "File") entries.push({ kind: "file", path, sha256: yield* hashFile(absolute), bytes: Number(stat.size) })
    else return yield* new AssertionFailure({ message: `Unexpected installed payload entry: ${path}` })
  })
  yield* visit("")
  if (!entries.some(entry => entry.kind === "file")) return yield* new AssertionFailure({ message: "Installed payload contains no files" })
  return InstalledPayload.make({ version: app.candidate.version, installer: Digest.make(app.candidate.artifact.sha256), entries })
}).pipe(Effect.mapError(error => error instanceof AssertionFailure ? error : new InfrastructureFailure({
  operation: "update-payload", message: "Could not inspect installed payload",
})))

/** Observation only: this never installs or repairs the updater's result. */
export const verifyInstalledPayload = (app: InstalledApplication, expected: typeof InstalledPayload.Type) => Effect.gen(function* () {
  const actual = yield* installedPayload(app)
  if (!Schema.equivalence(InstalledPayload)(actual, expected)) return yield* new AssertionFailure({
    message: "Updated payload differs from a clean native installation of the admitted candidate",
  })
  return PackagePayload.make({ version: actual.version, files: actual.entries.flatMap(entry => entry.kind === "file"
    ? [{ path: join(app.root, entry.path), sha256: entry.sha256, bytes: entry.bytes }] : []) })
})
