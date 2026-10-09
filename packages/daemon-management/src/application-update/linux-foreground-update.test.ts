import { CommandExecutor } from "@effect/platform"
import { Deferred, Effect, Fiber, Option, Ref } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../../release/src/hosted-update/release"
import { PreparedUpdateFailed, PreparedUpdateStore } from "../desktop-native/prepared-update"
import { TerminalCommand } from "../desktop-native/terminal-command"
import { completeLinuxForegroundUpdate } from "./linux-foreground-update"

const key = generateKeyPairSync("ed25519")
const release = await Effect.runPromise(signUpdateRelease({ version: "0.1.6", bytes: 1,
  sha256: createHash("sha256").update("x").digest("hex") }, { os: "linux", arch: "arm64", package: "deb" }, key.privateKey))
const fixture = (scenario: "success" | "absent" | "verify" | "attempt" | "installer" | "version", prompt = false, installerCode = 1) => {
  const events: string[] = []
  const step = (name: string) => Effect.sync(() => { events.push(name) })
  const failed = new PreparedUpdateFailed({ message: "Fixture refused" })
  const store = PreparedUpdateStore.of({
    read: Effect.succeed(scenario === "absent" ? Option.none() : Option.some({ release, installation: { _tag: "Unattempted" } })),
    verify: () => step("verify").pipe(Effect.zipRight(scenario === "verify" ? failed : Effect.succeed("archive"))),
    recordAttempt: () => step("attempt").pipe(Effect.zipRight(scenario === "attempt" ? failed : Effect.void)),
    recordFailure: (_, kind) => step(`failure:${kind}`), complete: () => step("complete"), discard: step("discard"), removeAbandonedTransfers: Effect.void, outcome: Effect.succeed(Option.none()), recordOutcome: () => Effect.void, markOutcomeReported: Effect.void,
    prepare: () => Effect.die("Installation cannot prepare a different download"),
  })
  const terminal = TerminalCommand.of({
    run: (executable, args, options) => step("installer").pipe(Effect.map(() => {
      expect(executable).toBe("/usr/bin/sudo")
      expect(args).toEqual([...(prompt ? [] : ["-n"]), "--", "/usr/lib/magnitude-desktop/resources/magnitude", "_install-application-update", "/profile/updates/update.json", "--parent-stdin"])
      expect(options.stdin).toBe("lifetime")
      return scenario === "installer" ? installerCode : 0
    })),
  })
  const executor = { ...CommandExecutor.makeExecutor(() => Effect.die("Unexpected start")),
    string: (command: import("@effect/platform/Command").Command) => step("version").pipe(Effect.map(() => {
      if (command._tag !== "StandardCommand") throw new Error("Expected version command")
      expect(command.command).toBe("/usr/lib/magnitude-desktop/resources/magnitude")
      expect(command.args).toEqual(["--version"])
      return scenario === "version" ? "0.1.5\n" : "0.1.6\n"
    })),
  }
  return { events, store, executor, terminal }
}
describe("foreground Linux update completion", () => {
  it.each([false, true])("waits for replacement and verifies its version before clearing state (prompt %s)", async prompt => {
    const f = fixture("success", prompt)
    expect(await Effect.runPromise(completeLinuxForegroundUpdate("/profile", prompt).pipe(
      Effect.provideService(PreparedUpdateStore, f.store), Effect.provideService(CommandExecutor.CommandExecutor, f.executor), Effect.provideService(TerminalCommand, f.terminal)))).toBe("0.1.6")
    expect(f.events).toEqual(["verify", "attempt", "installer", "version", "complete"])
  })
  it.each([
    ["absent", []], ["verify", ["verify", "failure:verify"]], ["attempt", ["verify", "attempt"]],
    ["installer", ["verify", "attempt", "installer", "failure:authorization"]],
    ["version", ["verify", "attempt", "installer", "version", "failure:install"]],
  ] as const)("retains the preparation after %s failure", async (scenario, events) => {
    const f = fixture(scenario)
    expect(await Effect.runPromise(completeLinuxForegroundUpdate("/profile", false).pipe(
      Effect.provideService(PreparedUpdateStore, f.store), Effect.provideService(CommandExecutor.CommandExecutor, f.executor), Effect.provideService(TerminalCommand, f.terminal), Effect.isFailure))).toBe(true)
    expect(f.events).toEqual(events)
  })
  it.each([[1, "authorization"], [3, "verify"], [4, "install"], [9, "install"]] as const)("classifies installer exit %i as %s", async (code, reason) => {
    const f = fixture("installer", false, code)
    expect(await Effect.runPromise(completeLinuxForegroundUpdate("/profile", false).pipe(
      Effect.provideService(PreparedUpdateStore, f.store), Effect.provideService(CommandExecutor.CommandExecutor, f.executor), Effect.provideService(TerminalCommand, f.terminal), Effect.isFailure))).toBe(true)
    expect(f.events.at(-1)).toBe(`failure:${reason}`)
  })
  it("does not clear the attempted record or continue after cancellation", async () => {
    const f = fixture("success")
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const entered = yield* Deferred.make<void>()
      const retired = yield* Ref.make(false)
      const worker = yield* completeLinuxForegroundUpdate("/profile", false).pipe(
        Effect.provideService(PreparedUpdateStore, f.store), Effect.provideService(CommandExecutor.CommandExecutor, f.executor),
        Effect.provideService(TerminalCommand, {
          run: () => Deferred.succeed(entered, undefined).pipe(Effect.zipRight(Effect.never), Effect.ensuring(Ref.set(retired, true))),
        }), Effect.forkScoped)
      yield* Deferred.await(entered)
      yield* Fiber.interrupt(worker)
      expect(yield* Ref.get(retired)).toBe(true)
      expect(f.events).toEqual(["verify", "attempt"])
    })))
  })
})
