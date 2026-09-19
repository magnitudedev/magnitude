import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { captureRetainedProfile, verifyRetainedProfile } from "../src/retained-profile"

for (const mutation of ["none", "contents", "delete", "directory", "link", "outside"] as const) test(`profile retention detects ${mutation} without following external links`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-retention-" })
  const profile = join(root, "profile")
  yield* fs.makeDirectory(join(profile, "empty"), { recursive: true })
  yield* fs.writeFileString(join(profile, "config.json"), '{"theme":"dark"}')
  const outside = join(root, "outside")
  yield* fs.writeFileString(outside, "outside original")
  yield* fs.symlink(outside, join(profile, "link"))
  const before = yield* captureRetainedProfile(profile)
  expect(before.entries.filter(entry => entry.kind === "file")).toHaveLength(1)
  if (mutation === "contents") yield* fs.writeFileString(join(profile, "config.json"), '{"theme":"lite"}')
  if (mutation === "delete") yield* fs.remove(profile, { recursive: true })
  if (mutation === "directory") yield* fs.remove(join(profile, "empty"), { recursive: true })
  if (mutation === "link") {
    yield* fs.remove(join(profile, "link"))
    yield* fs.symlink(join(root, "missing"), join(profile, "link"))
  }
  if (mutation === "outside") yield* fs.writeFileString(outside, "outside changed")
  const result = yield* verifyRetainedProfile(profile, before).pipe(Effect.either)
  expect(result._tag).toBe(mutation === "none" || mutation === "outside" ? "Right" : "Left")
  if (result._tag === "Left") expect(result.left._tag).toBe("AssertionFailure")
})).pipe(Effect.provide(BunContext.layer))))

test("an empty profile cannot establish retention", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-retention-empty-" })
  const result = yield* captureRetainedProfile(root).pipe(Effect.either)
  expect(result._tag).toBe("Left")
})).pipe(Effect.provide(BunContext.layer))))
