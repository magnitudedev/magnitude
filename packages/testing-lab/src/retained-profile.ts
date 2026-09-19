import { FileSystem } from "@effect/platform"
import type { PlatformError } from "@effect/platform/Error"
import { Effect, Option, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { AssertionFailure, Digest, InfrastructureFailure } from "./domain"

const ProfilePath = Schema.NonEmptyString.pipe(Schema.brand("RetainedProfilePath"))
const Entry = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("directory"), path: ProfilePath }),
  Schema.Struct({ kind: Schema.Literal("link"), path: ProfilePath, target: Schema.String }),
  Schema.Struct({ kind: Schema.Literal("file"), path: ProfilePath, bytes: Schema.Number, digest: Digest }),
)
export const RetainedProfile = Schema.Struct({ entries: Schema.Array(Entry) })
export type RetainedProfile = typeof RetainedProfile.Type

/** Capture after normal application shutdown. Hash contents without copying secrets or model bytes into evidence. */
export const captureRetainedProfile = (root: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const entries: (typeof Entry.Type)[] = []
  const walk = (relative: string): Effect.Effect<void, PlatformError | InfrastructureFailure> => Effect.gen(function* () {
    for (const name of (yield* fs.readDirectory(join(root, relative))).sort()) {
      if (entries.length >= 100_000) return yield* new InfrastructureFailure({ operation: "profile-retention", message: "Profile exceeds inspection entry limit" })
      const path = ProfilePath.make(relative ? `${relative}/${name}` : name)
      const absolute = join(root, path)
      const link = yield* fs.readLink(absolute).pipe(Effect.option)
      if (Option.isSome(link)) { entries.push({ kind: "link", path, target: link.value }); continue }
      const stat = yield* fs.stat(absolute)
      if (stat.type === "Directory") {
        entries.push({ kind: "directory", path })
        yield* walk(path)
      } else if (stat.type === "File") {
        const hash = createHash("sha256")
        let bytes = 0
        yield* fs.stream(absolute).pipe(Stream.runForEach(chunk => Effect.sync(() => { hash.update(chunk); bytes += chunk.byteLength })))
        if (BigInt(bytes) !== BigInt(stat.size)) return yield* new InfrastructureFailure({ operation: "profile-retention", message: `Profile changed during capture: ${path}` })
        entries.push({ kind: "file", path, bytes, digest: Digest.make(hash.digest("hex")) })
      } else return yield* new InfrastructureFailure({ operation: "profile-retention", message: `Profile contains an active or unsupported filesystem object: ${path}` })
    }
  })
  yield* walk("")
  if (!entries.some(entry => entry.kind === "file")) return yield* new InfrastructureFailure({ operation: "profile-retention", message: "An empty profile cannot qualify user-data retention" })
  return RetainedProfile.make({ entries })
})

export const verifyRetainedProfile = (root: string, before: RetainedProfile) => Effect.gen(function* () {
  const after = yield* captureRetainedProfile(root).pipe(Effect.mapError(error => error._tag === "SystemError" && error.reason === "NotFound"
    ? new AssertionFailure({ message: "Uninstall removed retained application data" }) : error))
  if (!Schema.equivalence(RetainedProfile)(before, after)) return yield* new AssertionFailure({ message: "Uninstall changed retained application data" })
  return after
})
