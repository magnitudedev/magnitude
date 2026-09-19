import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { expect, test } from "vitest"
import { DesktopDriver, type DesktopLaunch } from "../src/desktop-driver"
const events: string[] = []
const launchDriver = (config: DesktopLaunch) => Layer.scoped(DesktopDriver, Effect.acquireRelease(
  Effect.sync(() => { events.push(`open:${config.evidence}`); return {
    updates: { action: () => Effect.die("Unused"), automatic: () => Effect.die("Unused"), wait: () => Effect.die("Unused") },
    identity: () => Effect.die("Not used by this fixture"), host: () => Effect.succeed("test-version"),
    navigate: () => Effect.void, ready: () => Effect.void, serviceFailure: () => Effect.succeed("unused"), search: () => Effect.void,
    details: () => Effect.void, download: () => Effect.void, load: () => Effect.void,
    connect: () => Effect.void, connectionFailure: () => Effect.succeed("unused"), disconnect: () => Effect.void, theme: () => Effect.void,
    verifyTheme: () => Effect.void, screenshot: () => Effect.succeed("unused"),
    text: () => Effect.succeed("unused"), quit: () => Effect.void, chrome: () => Effect.void,
  } satisfies DesktopDriver }),
  () => Effect.sync(() => { events.push(`close:${config.evidence}`) }),
))
import { desktopSession } from "../src/desktop-session"
test("closes each app and trace before its replacement starts and closes the last on scope exit", async () => {
  events.length = 0
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const session = yield* desktopSession({ executable: "/unused", profile: "/profile", evidence: "/evidence", port: 11349, environment: {} }, () => {}, launchDriver)
    expect(events).toEqual([])
    yield* session.driver
    yield* session.driver
    expect(events).toEqual(["open:/evidence"])
    yield* session.restart
    yield* session.restart
  })).pipe(Effect.provide(BunContext.layer)))
  expect(events).toEqual(["open:/evidence", "close:/evidence", "open:/evidence/relaunch-1", "close:/evidence/relaunch-1", "open:/evidence/relaunch-2", "close:/evidence/relaunch-2"])
})
