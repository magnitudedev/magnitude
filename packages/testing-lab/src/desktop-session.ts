import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option, ScopedRef } from "effect"
import { join } from "node:path"
import { DesktopDriver, type DesktopLaunch, playwrightDesktop } from "./desktop-driver"

/** A restart releases the old process and its trace before acquiring its replacement. */
export const desktopSession = (config: DesktopLaunch, onCleanupError: (detail: string) => void, launchDriver = playwrightDesktop) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const current = yield* ScopedRef.make(() => Option.none<DesktopDriver>())
  let launch = 0
  const acquire = Effect.gen(function* () {
    const evidence = launch++ === 0 ? config.evidence : join(config.evidence, `relaunch-${launch - 1}`)
    const context = yield* Layer.build(launchDriver({ ...config, evidence }, undefined, onCleanupError))
    return Option.some(Context.get(context, DesktopDriver))
  }).pipe(Effect.provideService(FileSystem.FileSystem, fs))
  const driver = Effect.gen(function* () {
    const value = yield* ScopedRef.get(current)
    if (Option.isSome(value)) return value.value
    yield* ScopedRef.set(current, acquire)
    return Option.getOrThrow(yield* ScopedRef.get(current))
  })
  const stop = ScopedRef.set(current, Effect.succeed(Option.none<DesktopDriver>()))
  const restart = stop.pipe(Effect.zipRight(driver))
  return { driver, restart, stop }
})
