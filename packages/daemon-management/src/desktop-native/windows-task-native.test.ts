import { Effect, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { nativeWindowsJobOwnerLayer, nativeWindowsPrivatePipesLayer } from "@magnitudedev/utils/windows-native"
import { LegacyWindowsStartup, makeWindowsLegacyStartup } from "./legacy-startup-windows"
import { makeWindowsLegacyTaskControl } from "./windows-task-query"

// Only the fixture script creates this disabled registration and supplies these paths.
const addon = process.env.MAGNITUDE_TASK_FIXTURE_ADDON
const executable = process.env.MAGNITUDE_TASK_FIXTURE_HELPER
describe.skipIf(process.platform !== "win32" || !addon || !executable)("native Windows scheduled task retirement", () => {
  it("recognizes exported disabled registration, fences stale input, retires and replays", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const control = yield* makeWindowsLegacyTaskControl({ executable: executable!, environment: process.env })
      const startup = yield* makeWindowsLegacyStartup(control)
      const captured = yield* startup.inspect
      expect(Option.isSome(captured)).toBe(true)
      if (Option.isNone(captured)) return yield* Effect.die("Fixture registration is absent")
      expect(captured.value.enabled).toBe(false)
      expect(captured.value.executable).toBe(process.env.MAGNITUDE_TASK_FIXTURE_SERVICE)
      // Exercise the native fence directly, independently of the adapter's reread.
      const stale = LegacyWindowsStartup.make({ ...captured.value,
        digest: LegacyWindowsStartup.fields.digest.make("0".repeat(64)),
      })
      expect((yield* Effect.either(control.retire(stale)))._tag).toBe("Left")
      expect(yield* startup.inspect).toEqual(captured)
      const wrongUser = LegacyWindowsStartup.make({ ...captured.value,
        principalSid: LegacyWindowsStartup.fields.principalSid.make("S-1-5-21-0-0-0-999999"),
      })
      expect((yield* Effect.either(control.retire(wrongUser)))._tag).toBe("Left")
      expect(yield* startup.inspect).toEqual(captured)
      yield* startup.unregister(captured.value)
      expect(Option.isNone(yield* startup.inspect)).toBe(true)
      yield* control.retire(captured.value)
      yield* startup.unregister(captured.value)
    })).pipe(Effect.provide(Layer.merge(
      nativeWindowsJobOwnerLayer(addon!), nativeWindowsPrivatePipesLayer(addon!),
    ))))
  }, 60000)
})
