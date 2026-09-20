import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { expect, test } from "vitest"
import { WindowsRuntimePreparation, windowsRuntimePreparation, windowsDesktopReadiness } from "../src/providers/windows-initialization"

test("Windows preparation distinguishes diagnostic servers and validates download authority", () => {
  const valid = (value: unknown) => Schema.decodeUnknownEither(WindowsRuntimePreparation)(value)._tag === "Right"
  const config = { distribution: { os: "windows", version: "11" }, adminUsername: "labworker", architecture: "x64",
    runtime: { url: "https://example.com/runtime?sig=private-capability", sha256: "a".repeat(64), bytes: 100 } }
  expect(valid(config)).toBe(true)
  expect(valid({ ...config, distribution: { os: "windows-server", version: "2025" } })).toBe(true)
  for (const patch of [{ distribution: { os: "windows", version: "2025" } }, { architecture: "arm64" },
    { adminUsername: "labworker'; exit 0" }, { runtime: { ...config.runtime, url: "http://example.com/runtime" } },
    { runtime: { ...config.runtime, url: "https://user:secret@example.com/runtime" } }]) {
    expect(valid({ ...config, ...patch })).toBe(false)
  }
})

test("Windows runtime delivery uses a protected value rather than placing it in native arguments", () => Effect.runPromise(Effect.gen(function* () {
  const script = yield* windowsRuntimePreparation
  expect(script).toContain("$env:LAB_INITIALIZATION = $LAB_INITIALIZATION")
  expect(script).toContain("& $tools.powershell -NoProfile -NonInteractive -File $script")
  expect(script).not.toContain("-LAB_INITIALIZATION $LAB_INITIALIZATION")
  expect(script).not.toContain("?sig=")
  for (const [digest, user] of [["bad", "labworker"], ["a".repeat(64), "lab'; exit 0"]]) {
    expect((yield* windowsDesktopReadiness(digest!, user!).pipe(Effect.either))._tag).toBe("Left")
  }
}).pipe(Effect.provide(BunContext.layer))))
