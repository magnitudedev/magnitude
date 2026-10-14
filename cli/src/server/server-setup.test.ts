import { Duration, Effect, Either, Fiber, Option, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { portTaken, SERVER_START_TIMEOUT, type ServerSetupFailed, ServerSetupHost, serverRemove, serverSetup, WINDOWS_SERVER_MESSAGE } from "./server-setup"

const harness = (overrides: Partial<ServerSetupHost> = {}) => {
  const events: string[] = []
  const output: string[] = []
  const host: ServerSetupHost = {
    platform: "linux", user: "ada", isRoot: false, underSudo: false,
    hasServiceManager: Effect.succeed(true),
    personalOwner: Effect.succeed(Option.none()),
    isSetUp: Effect.succeed(true),
    hasRemnants: Effect.succeed(true),
    hasTerminal: Effect.succeed(true),
    sudoWithoutPrompt: Effect.succeed(false),
    servicePort: Effect.succeed("Service"),
    serviceRuns: Effect.succeed(Option.some(0)),
    serviceLog: Effect.succeed([]),
    confirm: () => Effect.sync(() => { events.push("confirm"); return true }),
    runRootStep: args => Effect.sync(() => { events.push(`root ${args.join(" ")}`) }),
    enableNetworkAccess: Effect.sync(() => { events.push("network"); return { addresses: ["192.168.1.5"], port: 10100, key: "mag-key" } }),
    write: text => Effect.sync(() => { output.push(text) }),
    ...overrides,
  }
  const run = (program: Effect.Effect<void, ServerSetupFailed, ServerSetupHost>) => Effect.runPromise(program.pipe(Effect.provideService(ServerSetupHost, host), Effect.either))
  return { events, host, output: () => output.join(""), run }
}

describe("magnitude server setup", () => {
  it("runs only the root step through sudo, then turns on network access and prints how to connect once", async () => {
    const h = harness()
    expect(Either.isRight(await h.run(serverSetup))).toBe(true)
    expect(h.events).toEqual(["root _server-install ada", "network"])
    expect(h.output()).toContain("http://192.168.1.5:10100")
    expect(h.output()).toContain("Network access key: mag-key")
    expect(h.output()).toContain("journalctl -u magnitude")
    expect(h.output()).toContain("magnitude status")
    expect(h.output()).not.toMatch(/magnitude group|log out/i)
  })
  it("refuses without systemd and points to `magnitude serve`", async () => {
    const h = harness({ hasServiceManager: Effect.succeed(false) })
    const result = await h.run(serverSetup)
    expect(Either.isLeft(result) && result.left.message).toContain("magnitude serve")
    expect(h.events).toEqual([])
  })
  it.each([["Desktop", "Quit the Magnitude desktop app"], ["Headless", "Stop `magnitude serve`"]] as const)("refuses while %s runs in the person's profile", async (owner, message) => {
    const h = harness({ personalOwner: Effect.succeed(Option.some(owner)), isSetUp: Effect.succeed(false) })
    const result = await h.run(serverSetup)
    expect(Either.isLeft(result) && result.left.message).toContain(message)
    expect(h.events).toEqual([])
  })
  it("finishes a macOS setup whose service is already serving, instead of asking to stop it", async () => {
    const h = harness({ platform: "darwin", personalOwner: Effect.succeed(Option.some("Headless")) })
    expect(Either.isRight(await h.run(serverSetup))).toBe(true)
    expect(h.events).toEqual(["root _server-install ada", "network"])
  })
  it("still refuses the desktop app when the service is registered", async () => {
    const h = harness({ platform: "darwin", personalOwner: Effect.succeed(Option.some("Desktop")) })
    expect(Either.isLeft(await h.run(serverSetup))).toBe(true)
    expect(h.events).toEqual([])
  })
  it.each([["linux", "journalctl -u magnitude"], ["darwin", "~/.magnitude/logs/server.log"]] as const)(
    "gives up on a %s service that never becomes ready, even mid-request, and says where its log is", async (platform, logs) => {
      const h = harness({ platform, enableNetworkAccess: Effect.uninterruptible(Effect.never) })
      const result = await Effect.runPromise(Effect.gen(function* () {
        const fiber = yield* Effect.fork(serverSetup.pipe(Effect.provideService(ServerSetupHost, h.host), Effect.either))
        yield* TestClock.adjust(SERVER_START_TIMEOUT)
        return yield* Fiber.join(fiber)
      }).pipe(Effect.provide(TestContext.TestContext)))
      expect(Either.isLeft(result) && result.left.message).toBe(`The Magnitude server did not become ready within 2 minutes.\nSee its log: ${logs}\n`
        + "Once it's fixed, run `magnitude server setup` again to finish, or `magnitude server remove` to undo setup.")
      expect(h.output()).toBe("Waiting for the Magnitude server to start…\n")
    })
  it("refuses before changing anything while another account's Magnitude or another program holds the port", async () => {
    const h = harness({ isSetUp: Effect.succeed(false), servicePort: Effect.succeed("Other") })
    const result = await h.run(serverSetup)
    expect(Either.isLeft(result) && result.left.message).toBe(`${portTaken} Nothing was changed.`)
    expect(h.events).toEqual([])
  })
  it("never configures another program that takes the port after the root step, and says what the log says", async () => {
    let checks = 0
    const h = harness({
      servicePort: Effect.sync(() => checks++ === 0 ? "Free" : "Other"),
      serviceLog: Effect.succeed(["Started magnitude.service.", "Error: Port 10100 is already in use.", "    at listen (/$bunfs/root/magnitude:1:2)", "Error: Port 10100 is already in use."]),
    })
    const result = await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* Effect.fork(serverSetup.pipe(Effect.provideService(ServerSetupHost, h.host), Effect.either))
      yield* TestClock.adjust(Duration.seconds(30))
      return yield* Fiber.join(fiber)
    }).pipe(Effect.provide(TestContext.TestContext)))
    expect(Either.isLeft(result) && result.left.message).toBe([portTaken, "Its log ends with:", "  Started magnitude.service.",
      "  Error: Port 10100 is already in use.", "See its log: journalctl -u magnitude",
      "Once it's fixed, run `magnitude server setup` again to finish, or `magnitude server remove` to undo setup."].join("\n"))
    expect(h.events).toEqual(["root _server-install ada"])
  })
  it("stops waiting for a service the service manager keeps restarting", async () => {
    let runs = 4
    const h = harness({ servicePort: Effect.succeed("Free"), serviceRuns: Effect.sync(() => Option.some(runs++)),
      serviceLog: Effect.succeed(["Magnitude couldn't download its inference engine (HTTP 404)."]) })
    const result = await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* Effect.fork(serverSetup.pipe(Effect.provideService(ServerSetupHost, h.host), Effect.either))
      yield* TestClock.adjust(Duration.seconds(10))
      return yield* Fiber.join(fiber)
    }).pipe(Effect.provide(TestContext.TestContext)))
    expect(Either.isLeft(result) && result.left.message).toContain("The Magnitude server keeps stopping as it starts.\nIts log ends with:\n  Magnitude couldn't download its inference engine (HTTP 404).\n")
    expect(h.events).toEqual(["root _server-install ada"])
  })
  it("fails at once without a terminal or passwordless sudo, instead of waiting at a prompt", async () => {
    const h = harness({ hasTerminal: Effect.succeed(false) })
    const result = await h.run(serverSetup)
    expect(Either.isLeft(result) && result.left.message).toBe("Server setup needs your password; run it in a terminal.")
    expect(h.events).toEqual([])
  })
  it("needs no terminal when sudo can run without asking, as after the install script's sudo -v", async () => {
    const h = harness({ hasTerminal: Effect.succeed(false), sudoWithoutPrompt: Effect.succeed(true) })
    expect(Either.isRight(await h.run(serverSetup))).toBe(true)
    expect(h.events).toEqual(["root _server-install ada", "network"])
  })
  it("sets up a root login's service directly", async () => {
    const h = harness({ isRoot: true, user: "root", hasTerminal: Effect.succeed(false) })
    expect(Either.isRight(await h.run(serverSetup))).toBe(true)
    expect(h.events).toEqual(["root _server-install", "network"])
    expect(h.output()).not.toContain("magnitude group")
  })
  it("refuses to run under sudo", async () => {
    const h = harness({ isRoot: true, underSudo: true })
    const result = await h.run(serverSetup)
    expect(Either.isLeft(result) && result.left.message).toContain("without sudo")
    expect(h.events).toEqual([])
  })
  it("fails with the Windows message and changes nothing", async () => {
    for (const program of [serverSetup, serverRemove]) {
      const h = harness({ platform: "win32" })
      const result = await h.run(program)
      expect(Either.isLeft(result) && result.left.message).toBe(WINDOWS_SERVER_MESSAGE)
      expect(h.output()).toBe("")
      expect(h.events).toEqual([])
    }
  })
  it("does not turn on network access when the root step fails", async () => {
    const h = harness({ runRootStep: () => Effect.fail({ _tag: "ServerSetupFailed", message: "denied" } as never) })
    expect(Either.isLeft(await h.run(serverSetup))).toBe(true)
    expect(h.events).toEqual([])
  })
})

describe("magnitude server remove", () => {
  it("asks for confirmation, then runs the root step and names the kept data", async () => {
    const h = harness()
    expect(Either.isRight(await h.run(serverRemove))).toBe(true)
    expect(h.events).toEqual(["confirm", "root _server-remove"])
    expect(h.output()).toContain("/var/lib/magnitude")
  })
  it("changes nothing when the person declines", async () => {
    const h = harness({ confirm: () => Effect.succeed(false) })
    expect(Either.isRight(await h.run(serverRemove))).toBe(true)
    expect(h.events).toEqual([])
    expect(h.output()).toBe("Nothing was changed.\n")
  })
  it("cleans up what a partial setup or removal left, even without the marker", async () => {
    const h = harness({ isSetUp: Effect.succeed(false) })
    expect(Either.isRight(await h.run(serverRemove))).toBe(true)
    expect(h.events).toEqual(["confirm", "root _server-remove"])
    expect(h.output()).toContain("Its data is still in /var/lib/magnitude, now owned by root.")
  })
  it("reports when nothing of a server exists", async () => {
    const h = harness({ isSetUp: Effect.succeed(false), hasRemnants: Effect.succeed(false) })
    expect(Either.isRight(await h.run(serverRemove))).toBe(true)
    expect(h.events).toEqual([])
  })
  it("keeps ~/.magnitude on macOS", async () => {
    const h = harness({ platform: "darwin" })
    expect(Either.isRight(await h.run(serverRemove))).toBe(true)
    expect(h.output()).toContain("~/.magnitude")
  })
})
