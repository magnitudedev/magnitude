import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Option, Schema } from "effect"
import { createHash } from "node:crypto"
import { join, resolve } from "node:path"
import { Digest, InvalidInput } from "../src/domain"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { extractSource, snapshotSource } from "../src/snapshot"
import { Stream } from "effect"

/** Publish this trusted runtime separately from any candidate submitted to the lab. */
const main = Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const [source, destination] = process.argv.slice(2)
  if (!source || !destination || process.argv.length !== 4) return yield* new InvalidInput({ message: "Expected source checkout and a new output directory" })
  const fs = yield* FileSystem.FileSystem
  const output = resolve(destination)
  yield* fs.makeDirectory(output, { mode: 0o700 })
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "lab-runtime-" })
  const objects = join(temporary, "objects")
  const snapshot = yield* snapshotSource(resolve(source), objects)
  yield* extractSource(snapshot.manifest, objects, join(temporary, "runtime"))
  const archive = join(output, "runtime.tar.gz")
  yield* checkedCommand("tar", [...(process.platform === "darwin" ? ["--no-xattrs"] : []), "-czf", archive, "-C", temporary, "runtime"],
    { timeoutMs: 600_000, cwd: Option.none(), env: { COPYFILE_DISABLE: "1" } })
  const hash = createHash("sha256")
  yield* fs.stream(archive).pipe(Stream.runForEach(bytes => Effect.sync(() => { hash.update(bytes) })))
  const Receipt = Schema.Struct({ schemaVersion: Schema.Literal(1), sourceDigest: Digest, sha256: Digest, bytes: Schema.Number,
    archive: Schema.String, sourceCommit: Schema.String })
  const receipt = { schemaVersion: 1 as const, sourceDigest: snapshot.digest, sha256: Digest.make(hash.digest("hex")),
    bytes: Number((yield* fs.stat(archive)).size), archive, sourceCommit: snapshot.manifest.commit }
  const json = yield* Schema.encode(Schema.parseJson(Receipt))(receipt)
  yield* fs.writeFileString(join(output, "runtime.json"), json, { mode: 0o600, flag: "wx" })
  yield* Console.log(json)
}))
BunRuntime.runMain(main.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
