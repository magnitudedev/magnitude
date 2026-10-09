import { Effect, Either, Option } from "effect"
import { describe, expect, it } from "vitest"
import { ServerSetupHost, serverRemove, serverSetup, WINDOWS_SERVER_MESSAGE } from "./server-setup"

const harness = (overrides: Partial<ServerSetupHost> = {}) => {
  const events: string[] = []
  const output: string[] = []
  const host: ServerSetupHost = {
    platform: "linux", user: "ada", isRoot: false,
    hasServiceManager: Effect.succeed(true),
    personalOwner: Effect.succeed(Option.none()),
    isSetUp: Effect.succeed(true),
    hasTerminal: Effect.succeed(true),
    sudoWithoutPrompt: Effect.succeed(false),
    confirm: () => Effect.sync(() => { events.push("confirm"); return true }),
    runRootStep: args => Effect.sync(() => { events.push(`root ${args.join(" ")}`) }),
    enableNetworkAccess: Effect.sync(() => { events.push("network"); return { addresses: ["192.168.1.5"], port: 10100, key: "mag-key" } }),
    write: text => Effect.sync(() => { output.push(text) }),
    ...overrides,
  }
  const run = (program: typeof serverSetup) => Effect.runPromise(program.pipe(Effect.provideService(ServerSetupHost, host), Effect.either))
  return { events, output: () => output.join(""), run }
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
  })
  it("refuses without systemd and points to `magnitude serve`", async () => {
    const h = harness({ hasServiceManager: Effect.succeed(false) })
    const result = await h.run(serverSetup)
    expect(Either.isLeft(result) && result.left.message).toContain("magnitude serve")
    expect(h.events).toEqual([])
  })
  it.each([["Desktop", "Quit the Magnitude desktop app"], ["Headless", "Stop `magnitude serve`"]] as const)("refuses while %s runs in the person's profile", async (owner, message) => {
    const h = harness({ personalOwner: Effect.succeed(Option.some(owner)) })
    const result = await h.run(serverSetup)
    expect(Either.isLeft(result) && result.left.message).toContain(message)
    expect(h.events).toEqual([])
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
  it("refuses to run under sudo", async () => {
    const h = harness({ isRoot: true })
    const result = await h.run(serverSetup)
    expect(Either.isLeft(result) && result.left.message).toContain("without sudo")
    expect(h.events).toEqual([])
  })
  it("prints the Windows message and changes nothing", async () => {
    for (const program of [serverSetup, serverRemove]) {
      const h = harness({ platform: "win32" })
      expect(Either.isRight(await h.run(program))).toBe(true)
      expect(h.output()).toBe(`${WINDOWS_SERVER_MESSAGE}\n`)
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
  it("reports when no server is set up", async () => {
    const h = harness({ isSetUp: Effect.succeed(false) })
    expect(Either.isRight(await h.run(serverRemove))).toBe(true)
    expect(h.events).toEqual([])
  })
  it("keeps ~/.magnitude on macOS", async () => {
    const h = harness({ platform: "darwin" })
    expect(Either.isRight(await h.run(serverRemove))).toBe(true)
    expect(h.output()).toContain("~/.magnitude")
  })
})
