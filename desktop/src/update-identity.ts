import { FileSystem } from "@effect/platform"
import { Context, Effect, Schema } from "effect"
import { createPrivateKey, generateKeyPairSync } from "node:crypto"
import { join } from "node:path"
import { signUpdateRequest, type UpdateSigningFailed } from "@magnitudedev/release/hosted-update"

export class UpdateIdentityFailed extends Schema.TaggedError<UpdateIdentityFailed>()("UpdateIdentityFailed", {}) {}
export interface UpdateIdentity {
  readonly sign: (url: URL) => Effect.Effect<string, UpdateSigningFailed>
}
export const UpdateIdentity = Context.GenericTag<UpdateIdentity>("desktop/UpdateIdentity")

/** Called after native application ownership. Only the desktop owner creates this installation key. */
export const makeUpdateIdentity = (clientStateDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = join(clientStateDirectory, "updates")
  const path = join(directory, "installation-key.pem")
  yield* fs.makeDirectory(directory, { recursive: true, mode: 0o700 })
  if (!(yield* fs.exists(path))) {
    const pem = yield* Effect.try({ try: () => generateKeyPairSync("ed25519").privateKey.export({ type: "pkcs8", format: "pem" }).toString(), catch: () => new UpdateIdentityFailed() })
    yield* Effect.scoped(Effect.gen(function* () {
      const temporary = yield* fs.makeTempDirectoryScoped({ directory, prefix: "identity-" })
      const pending = join(temporary, "key.pem")
      yield* fs.writeFileString(pending, pem, { mode: 0o600, flag: "wx" })
      yield* fs.rename(pending, path)
    })).pipe(Effect.uninterruptible)
  }
  const stat = yield* fs.stat(path)
  if (stat.type !== "File" || stat.size > 4096) return yield* new UpdateIdentityFailed()
  if (process.platform !== "win32") yield* fs.chmod(path, 0o600)
  const pem = yield* fs.readFileString(path)
  // Invalid persisted material is an error, never an excuse to silently create a new identity.
  const key = yield* Effect.try({ try: () => {
    const key = createPrivateKey(pem)
    if (key.asymmetricKeyType !== "ed25519") throw new Error("Wrong key type")
    return key
  }, catch: () => new UpdateIdentityFailed() })
  return UpdateIdentity.of({ sign: url => signUpdateRequest(key, url) })
}).pipe(Effect.mapError(() => new UpdateIdentityFailed()))
