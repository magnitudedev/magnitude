import { FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
import { join } from "node:path"
import { Candidate } from "../candidate"
import { AssertionFailure } from "../domain"
import { Installer } from "../installer"

/** Keep the admitted installer intact; corrupt a same-length private copy before native install. */
export const rejectCorruptInstaller = (candidate: Candidate) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const installer = yield* Installer
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-corrupt-package-" })
  const path = join(root, candidate.artifact.filename)
  yield* fs.copyFile(candidate.path, path)
  yield* Effect.scoped(Effect.gen(function* () {
    const file = yield* fs.open(path, { flag: "r+" })
    const first = yield* file.readAlloc(1)
    if (Option.isNone(first)) return yield* new AssertionFailure({ message: "Cannot exercise corruption with an empty installer" })
    yield* file.seek(0, "start")
    yield* file.write(new Uint8Array([first.value[0]! ^ 0xff]))
  }))
  const result = yield* installer.install({ ...candidate, path }).pipe(Effect.either)
  if (result._tag === "Right") {
    yield* installer.uninstall(result.right)
    return yield* new AssertionFailure({ message: "Native installation accepted an installer with altered bytes" })
  }
  if (result.left._tag !== "AssertionFailure" || result.left.message !== "Installer changed after download; refusing installation") return yield* result.left
}))
