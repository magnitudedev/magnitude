import { Effect, Layer, Option, Schema } from "effect"
import { expect, test } from "vitest"
import { installationSession } from "../src/installation-session"
import { Candidate } from "../src/candidate"
import { targets } from "../src/catalog"
import { Installer, InstalledApplication } from "../src/installer"
import { AssertionFailure } from "../src/domain"

const target = targets.find(target => target.id === "macos-15-arm64-metal-apple-silicon")!
const candidate = Schema.decodeUnknownSync(Candidate)({ version: "0.1.3", target, path: "/candidate.dmg",
  artifact: { id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64", filename: "Magnitude.dmg", bytes: 1, sha256: "a".repeat(64) } })
for (const scenario of ["reinstall", "remove", "remove-fails"] as const) test(`tracks installation ownership for ${scenario}`, async () => {
  let installs = 0, removals = 0
  const cleanupErrors: string[] = []
  const installer = Layer.succeed(Installer, {
    install: () => Effect.sync(() => { installs++; return InstalledApplication.make({ candidate, root: "/app", executable: "/app/exe", cli: "/app/cli", packageVersion: "0.1.3" }) }),
    uninstall: () => Effect.suspend(() => { removals++; return scenario === "remove-fails" ? Effect.fail(new AssertionFailure({ message: "Removal failed" })) : Effect.void }),
  })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const session = yield* installationSession(candidate, detail => { cleanupErrors.push(detail) })
    yield* Effect.all([session.get, session.get], { concurrency: "unbounded" })
    expect(installs).toBe(1)
    const removed = yield* session.remove.pipe(Effect.either)
    expect(removed._tag).toBe(scenario === "remove-fails" ? "Left" : "Right")
    if (scenario === "reinstall") yield* session.get
  })).pipe(Effect.provide(installer)))
  expect(installs).toBe(scenario === "reinstall" ? 2 : 1)
  expect(removals).toBe(scenario === "remove" ? 1 : 2)
  expect(cleanupErrors).toEqual(scenario === "remove-fails" ? ["Removal failed"] : [])
})
