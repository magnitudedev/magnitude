import { FileSystem } from "@effect/platform"
import { Effect, Stream } from "effect"
import { isAbsolute, resolve, sep } from "node:path"
import { ArtifactStore } from "./artifact-store"
import { Evidence, InfrastructureFailure } from "./domain"
import { sha256 } from "./snapshot"

const fail = (message: string) => new InfrastructureFailure({ operation: "evidence-export", message })
/** Export only an explicitly selected, closed diagnostic inside the owned evidence root. */
export const publishEvidenceFile = (root: string, relative: string, maxBytes: number) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const objects = yield* ArtifactStore
  if (isAbsolute(relative) || relative.split(/[\\/]/).some(part => part === ".." || part === "." || !part)) return yield* fail("Invalid evidence path")
  const canonicalRoot = yield* fs.realPath(root)
  const path = resolve(canonicalRoot, relative)
  if (!path.startsWith(canonicalRoot + sep) || (yield* fs.realPath(path)) !== path) return yield* fail("Evidence must be an owned file without symbolic links")
  const stat = yield* fs.stat(path)
  if (stat.type !== "File" || Number(stat.size) > maxBytes) return yield* fail("Evidence is not a bounded regular file")
  let size = 0
  const chunks = yield* fs.stream(path).pipe(Stream.tap(bytes => Effect.gen(function* () {
    size += bytes.byteLength
    if (size > maxBytes) return yield* fail("Evidence exceeded its byte limit during export")
  })), Stream.runCollect)
  if (size !== Number(stat.size)) return yield* fail("Evidence changed during export")
  const bytes = Buffer.concat(Array.from(chunks))
  const digest = sha256(bytes)
  yield* objects.put(digest, Stream.make(bytes))
  return Evidence.make({ path: `evidence/${relative.replaceAll("\\", "/")}`, sha256: digest, bytes: size })
}).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : fail(error.message)))
