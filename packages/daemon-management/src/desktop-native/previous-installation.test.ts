import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { PreviousInstallation, PreviousInstallationFailed, PreviousInstallationJournal, makePreviousInstallationUpgrade, type PreviousInstallationPlan } from "./previous-installation"

const plan: PreviousInstallationPlan = { _tag: "Unix", startup: Option.none(), tree: Option.none() }
const fixture = () => {
  let pending = Option.none<PreviousInstallationPlan>()
  let installed = true
  let failAt: "write" | "retire" | "clear" | undefined
  const events: string[] = []
  const fail = (stage: string) => new PreviousInstallationFailed({ message: stage })
  const installation = PreviousInstallation.of({
    inspect: Effect.sync(() => { events.push("inspect"); return installed ? Option.some(plan) : Option.none() }),
    retire: value => Effect.gen(function* () {
      expect(value).toEqual(plan)
      expect(Option.isSome(pending)).toBe(true)
      events.push("retire")
      if (failAt === "retire") return yield* fail("retire")
      installed = false
    }),
  })
  const journal = PreviousInstallationJournal.of({
    read: Effect.sync(() => { events.push("read"); return pending }),
    write: value => Effect.gen(function* () { events.push("write"); if (failAt === "write") return yield* fail("write"); pending = Option.some(value) }),
    clear: Effect.gen(function* () { events.push("clear"); if (failAt === "clear") return yield* fail("clear"); pending = Option.none() }),
  })
  const make = () => Effect.runPromise(makePreviousInstallationUpgrade.pipe(Effect.provideService(PreviousInstallation, installation), Effect.provideService(PreviousInstallationJournal, journal)))
  return { make, events, pending: () => pending, setInstalled: (value: boolean) => { installed = value }, failAt: (value: typeof failAt) => { failAt = value } }
}

describe("automatic previous installation upgrade", () => {
  it("persists identities before retirement and verifies absence afterward", async () => {
    const f = fixture(); await Effect.runPromise(await f.make())
    expect(f.events).toEqual(["read", "inspect", "write", "retire", "clear", "inspect"])
    expect(Option.isNone(f.pending())).toBe(true)
  })
  it("does not create a recovery record on a clean installation", async () => {
    const f = fixture(); f.setInstalled(false); await Effect.runPromise(await f.make())
    expect(f.events).toEqual(["read", "inspect"])
  })
  it("does not stop anything when recovery information cannot be saved", async () => {
    const f = fixture(); f.failAt("write")
    await expect(Effect.runPromise(await f.make())).rejects.toThrow("write")
    expect(f.events).not.toContain("retire")
  })
  it.each(["retire", "clear"] as const)("resumes after interruption at %s", async stage => {
    const f = fixture(); f.failAt(stage)
    await expect(Effect.runPromise(await f.make())).rejects.toThrow(stage)
    expect(Option.isSome(f.pending())).toBe(true)
    f.failAt(undefined); f.events.length = 0
    await Effect.runPromise(await f.make())
    expect(f.events.slice(0, 3)).toEqual(["read", "retire", "clear"])
    expect(Option.isNone(f.pending())).toBe(true)
  })
  it("checks again on later launches instead of trusting a permanent completion marker", async () => {
    const f = fixture(); const upgrade = await f.make()
    await Effect.runPromise(upgrade)
    f.setInstalled(true); f.events.length = 0
    await Effect.runPromise(upgrade)
    expect(f.events).toContain("retire")
  })
  it("serializes simultaneous startup attempts", async () => {
    const f = fixture(); const upgrade = await f.make()
    await Effect.runPromise(Effect.all([upgrade, upgrade], { concurrency: "unbounded" }))
    expect(f.events.filter(event => event === "retire")).toHaveLength(1)
  })
})
