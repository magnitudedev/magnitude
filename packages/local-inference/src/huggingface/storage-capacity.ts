import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { Effect, Layer } from "effect"
import { StorageCapacity, type StorageCapacityApi } from "./contracts"
import { StorageCapacityError } from "./errors"

export const makeStorageCapacity = (): Effect.Effect<StorageCapacityApi, never, CommandExecutor.CommandExecutor> => Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  return {
    availableBytes: (path) => {
      if (process.platform === "win32") return Effect.fail(new StorageCapacityError({ path, diagnostic: "UnsupportedPlatform" }))
      return Command.string(Command.make("df", "-Pk", path)).pipe(
        Effect.provideService(CommandExecutor.CommandExecutor, executor),
        Effect.mapError(() => new StorageCapacityError({ path, diagnostic: "CommandFailed" })),
        Effect.flatMap((output) => {
          const columns = output.trim().split("\n").at(-1)?.trim().split(/\s+/)
          const availableKiB = Number(columns?.at(-3))
          return Number.isSafeInteger(availableKiB) && availableKiB >= 0
            ? Effect.succeed(availableKiB * 1024)
            : Effect.fail(new StorageCapacityError({ path, diagnostic: "InvalidCommandOutput" }))
        }),
      )
    },
  }
})

export const StorageCapacityLive: Layer.Layer<StorageCapacity, never, CommandExecutor.CommandExecutor> =
  Layer.effect(StorageCapacity, makeStorageCapacity())
