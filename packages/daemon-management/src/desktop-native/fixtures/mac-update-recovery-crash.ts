import { Effect, Option, Schema } from "effect"
import { fileURLToPath } from "node:url"
import { MacUpdateFilesystem, nativeMacUpdateFilesystem } from "../mac-update-filesystem"
import { MacUpdateJournal } from "../mac-update-recovery"

const Phase = Schema.Literal("BeforeExchange", "AfterExchange", "AfterCommit", "BeforeRestore", "AfterRestore")
const crash = Effect.sync(() => process.kill(process.pid, "SIGKILL"))
const program = Effect.gen(function* () {
  const [root, stagingPath, phase] = yield* Schema.decodeUnknown(Schema.Tuple(Schema.NonEmptyString, Schema.NonEmptyString, Phase))(process.argv.slice(2))
  const fs = yield* MacUpdateFilesystem
  const installed = yield* fs.open(root, false)
  const staging = yield* fs.open(stagingPath, true)
  const bytes = Option.getOrThrow(yield* fs.readRecord(staging))
  const record = yield* Schema.decodeUnknown(Schema.parseJson(MacUpdateJournal))(new TextDecoder().decode(bytes))
  const transaction = record.transaction
  const write = (tag: MacUpdateJournal["_tag"]) => Schema.decodeUnknown(MacUpdateJournal)({ ...record, _tag: tag }).pipe(
    Effect.flatMap(Schema.encode(Schema.parseJson(MacUpdateJournal))), Effect.flatMap(text => fs.writeRecord(staging, Buffer.from(text))))
  yield* write("ExchangeIntent")
  if (phase === "BeforeExchange") return yield* crash
  yield* fs.exchange(installed, "Magnitude.app", transaction.previous.identity, staging, "Magnitude.app", transaction.replacement.identity)
  if (phase === "AfterExchange") return yield* crash
  if (phase === "AfterCommit") {
    yield* write("Committed")
    return yield* crash
  }
  yield* write("RestoreIntent")
  if (phase === "BeforeRestore") return yield* crash
  yield* fs.exchange(installed, "Magnitude.app", transaction.replacement.identity, staging, "Magnitude.app", transaction.previous.identity)
  return yield* crash
})
Effect.runPromise(program.pipe(Effect.scoped, Effect.provide(nativeMacUpdateFilesystem(
  fileURLToPath(new URL(`../../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url)),
)))).catch(error => { console.error(error); process.exitCode = 1 })
