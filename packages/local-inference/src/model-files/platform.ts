import { createHash } from "node:crypto"
import type { PlatformError } from "@effect/platform/Error"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Stream } from "effect"
import { Sha256Digest } from "./identity"
import type { FileSystemFailureReason } from "./types"

export const normalizeFileSystemFailure = (error: PlatformError): FileSystemFailureReason => {
  if (error._tag === "BadArgument") return "bad-argument"
  switch (error.reason) {
    case "NotFound": return "not-found"
    case "AlreadyExists": return "already-exists"
    case "PermissionDenied": return "permission-denied"
    case "InvalidData": return "invalid-data"
    case "BadResource": return "bad-resource"
    case "Busy": return "busy"
    case "TimedOut": return "timed-out"
    case "UnexpectedEof": return "unexpected-eof"
    case "Unknown": return "system-unknown"
    case "WouldBlock": return "would-block"
    case "WriteZero": return "write-zero"
  }
}

export const sha256File = (path: string): Effect.Effect<Sha256Digest, PlatformError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const hash = createHash("sha256")
    yield* fs.stream(path).pipe(Stream.runForEach((bytes) => Effect.sync(() => hash.update(bytes))))
    return Sha256Digest.make(hash.digest("hex"))
  })
