import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Option, Schema } from "effect"
import { ConnectionTransaction } from "./transaction"
import { writeFileAtomic as writeFileAtomicRaw } from "../utils/atomic-file"

class ConfigurationChanged extends Schema.TaggedError<ConfigurationChanged>()("ConfigurationChanged", { path: Schema.String }) {
  override get message() { return `Preserved concurrently changed configuration: ${this.path}` }
}

/** Replace a user-owned configuration file without exposing partial contents. */
const changeFile = (file: string, contents: Option.Option<string>) => Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
  const transaction = yield* Effect.serviceOption(ConnectionTransaction)
  const fs = yield* FileSystem.FileSystem
  if (Option.isSome(transaction)) {
    const before = yield* fs.readFileString(file).pipe(
      Effect.map(Option.some), Effect.catchTag("SystemError", (e) => e.reason === "NotFound" ? Effect.succeed(Option.none<string>()) : Effect.fail(e)),
    )
    yield* transaction.value.compensate(file, Effect.gen(function* () {
      const current = yield* fs.readFileString(file).pipe(
        Effect.map(Option.some), Effect.catchTag("SystemError", (e) => e.reason === "NotFound" ? Effect.succeed(Option.none<string>()) : Effect.fail(e)),
      )
      if (Option.getEquivalence<string>((a, b) => a === b)(current, before)) return
      if (!Option.getEquivalence<string>((a, b) => a === b)(current, contents)) return yield* new ConfigurationChanged({ path: file })
      if (Option.isSome(before)) yield* writeFileAtomicRaw(file, before.value)
      else yield* fs.remove(file)
    }))
  }
  yield* restore(Option.match(contents, {
    onSome: (contents) => writeFileAtomicRaw(file, contents),
    onNone: () => fs.remove(file).pipe(Effect.catchTag("SystemError", (e) => e.reason === "NotFound" ? Effect.void : Effect.fail(e))),
  }))
}))

export const writeFileAtomic = (file: string, contents: string) => changeFile(file, Option.some(contents))
export const removeConfigurationFile = (file: string) => changeFile(file, Option.none())
