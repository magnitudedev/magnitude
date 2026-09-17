import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { makeMacLegacyStartup } from "./legacy-startup-macos"
import { LegacyStartupCommands, NativeLegacyStartupCommands } from "./legacy-startup-command"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"

let home: string
let path: string
beforeEach(async () => {
  home = await mkdtemp(join(tmpdir(), "magnitude-legacy-login-"))
  path = join(home, "Library/LaunchAgents/dev.magnitude.acn.plist")
  await mkdir(join(home, "Library/LaunchAgents"), { recursive: true })
})
afterEach(() => rm(home, { recursive: true, force: true }))
const run = Effect.runPromise
const fixture = async (options: { loaded?: boolean; pid?: number; disabled?: boolean; plistDisabled?: boolean; noOverride?: boolean; wrongPath?: boolean; failDisable?: boolean } = {}) => {
  await writeFile(path, "verified fixture bytes")
  let loaded = options.loaded ?? false
  let pid = options.pid
  const calls: string[][] = []
  const commands = LegacyStartupCommands.of({ run: (executable, args) => Effect.sync(() => {
    calls.push([executable, ...args])
    const ok = (stdout = "") => ({ code: 0, stdout, stderr: "" })
    if (executable === "/usr/bin/plutil") return ok(JSON.stringify({ Label: "dev.magnitude.acn", ProgramArguments: ["/fixture/magnitude-service", "serve"], RunAtLoad: true, Disabled: options.plistDisabled ?? false }))
    if (args[0] === "print") return loaded ? ok(`path = ${options.wrongPath ? "/unrelated/file" : path}\n${pid === undefined ? "" : `pid = ${pid}\n`}`) : { code: 113, stdout: "", stderr: "not found" }
    if (args[0] === "print-disabled") return ok(options.noOverride ? "disabled services = {}" : `disabled services = {\n"dev.magnitude.acn" => ${options.disabled ? "disabled" : "enabled"}\n}`)
    if (args[0] === "disable") return options.failDisable ? { code: 1, stdout: "", stderr: "permission denied" } : ok()
    if (args[0] === "bootout") { loaded = false; return ok() }
    throw new Error(`Unexpected startup command: ${args.join(" ")}`)
  }) })
  const adapter = await run(makeMacLegacyStartup(home).pipe(Effect.provideService(LegacyStartupCommands, commands)))
  return { adapter, calls, setPid: (next: number) => { pid = next; loaded = true } }
}

describe.skipIf(process.platform === "win32")("legacy macOS startup registration", () => {
  it("reads an enabled but unloaded registration without mutation", async () => {
    const f = await fixture()
    expect(Option.getOrThrow(await run(f.adapter.inspect)).enabled).toBe(true)
    expect(f.calls.some(call => call.includes("disable") || call.includes("bootout"))).toBe(false)
  })
  it("preserves the disabled login preference", async () => {
    const f = await fixture({ disabled: true })
    expect(Option.getOrThrow(await run(f.adapter.inspect)).enabled).toBe(false)
  })
  it("respects plist Disabled unless launchctl explicitly overrides it", async () => {
    const disabled = await fixture({ plistDisabled: true, noOverride: true })
    expect(Option.getOrThrow(await run(disabled.adapter.inspect)).enabled).toBe(false)
    const overridden = await fixture({ plistDisabled: true })
    expect(Option.getOrThrow(await run(overridden.adapter.inspect)).enabled).toBe(true)
  })
  it("disables and unloads before removing the exact inspected file", async () => {
    const f = await fixture({ loaded: true })
    const snapshot = Option.getOrThrow(await run(f.adapter.inspect))
    await run(f.adapter.unregister(snapshot))
    expect(f.calls.filter(call => ["disable", "bootout"].includes(call[1]!)).map(call => call[1])).toEqual(["disable", "bootout"])
    await expect(readFile(path)).rejects.toMatchObject({ code: "ENOENT" })
    await run(f.adapter.unregister(snapshot))
    expect(f.calls.filter(call => call[1] === "disable")).toHaveLength(1)
  })
  it("removes an unloaded registration without bootout", async () => {
    const f = await fixture()
    await run(f.adapter.unregister(Option.getOrThrow(await run(f.adapter.inspect))))
    expect(f.calls.some(call => call[1] === "bootout")).toBe(false)
  })
  it("preserves a file changed after migration preparation", async () => {
    const f = await fixture()
    const snapshot = Option.getOrThrow(await run(f.adapter.inspect))
    await writeFile(path, "user changed it")
    await expect(run(f.adapter.unregister(snapshot))).rejects.toThrow("changed after migration was prepared")
    expect(await readFile(path, "utf8")).toBe("user changed it")
    expect(f.calls.some(call => call[1] === "disable")).toBe(false)
  })
  it("rejects a loaded job from another source path", async () => {
    const f = await fixture({ loaded: true, wrongPath: true })
    await expect(run(f.adapter.inspect)).rejects.toThrow("unverified source path")
    expect(f.calls.some(call => call[1] === "disable")).toBe(false)
  })
  it("retains the file when disabling fails", async () => {
    const f = await fixture({ failDisable: true })
    const snapshot = Option.getOrThrow(await run(f.adapter.inspect))
    await expect(run(f.adapter.unregister(snapshot))).rejects.toThrow("permission denied")
    expect(await readFile(path, "utf8")).toBe("verified fixture bytes")
    expect(f.calls.some(call => call[1] === "bootout")).toBe(false)
  })
  it("refuses to unload a replacement job process", async () => {
    const f = await fixture({ loaded: true, pid: 1234 })
    const snapshot = Option.getOrThrow(await run(f.adapter.inspect))
    f.setPid(1235)
    await expect(run(f.adapter.unregister(snapshot))).rejects.toThrow("different process")
    expect(f.calls.some(call => call[1] === "disable" || call[1] === "bootout")).toBe(false)
    expect(await readFile(path, "utf8")).toBe("verified fixture bytes")
  })
  it.skipIf(process.platform !== "darwin" || process.env.MAGNITUDE_TEST_LEGACY_LOGIN !== "1")("unregisters a real isolated launch agent", async () => {
    const nativeHome = await realpath(home)
    const label = `dev.magnitude.migration-test-${process.pid}`
    const target = `gui/${process.getuid!()}/${label}`
    const agent = join(nativeHome, `Library/LaunchAgents/${label}.plist`)
    const executable = join(nativeHome, "magnitude-service")
    await writeFile(executable, "#!/bin/sh\nexec /bin/sleep 120\n", { mode: 0o755 })
    await writeFile(agent, `<?xml version="1.0"?><plist version="1.0"><dict><key>Label</key><string>${label}</string><key>ProgramArguments</key><array><string>${executable}</string><string>serve</string></array><key>RunAtLoad</key><true/></dict></plist>`)
    const commands = await run(Effect.serviceOption(LegacyStartupCommands).pipe(Effect.provide(NativeLegacyStartupCommands), Effect.map(Option.getOrThrow)))
    const launchctl = (args: readonly string[]) => run(commands.run("/bin/launchctl", args))
    try {
      expect((await launchctl(["enable", target])).code).toBe(0)
      expect((await launchctl(["bootstrap", `gui/${process.getuid!()}`, agent])).code).toBe(0)
      const job = await launchctl(["print", target])
      const pid = Number(job.stdout.match(/^\s*pid = (\d+)$/m)?.[1])
      expect(Number.isSafeInteger(pid) && pid > 0).toBe(true)
      const identity = Option.getOrThrow(await run(ProcessGroupControllerLive.inspect(pid)))
      const adapter = await run(makeMacLegacyStartup(nativeHome, label).pipe(Effect.provide(NativeLegacyStartupCommands)))
      const snapshot = Option.getOrThrow(await run(adapter.inspect))
      expect(snapshot.enabled).toBe(true)
      await run(adapter.unregister(snapshot))
      expect((await launchctl(["print", target])).code).toBe(113)
      expect(await run(ProcessGroupControllerLive.waitForGroupExit({ leader: identity }, "3 seconds"))).toBe(true)
      await expect(readFile(agent)).rejects.toMatchObject({ code: "ENOENT" })
      expect(Option.isNone(await run(adapter.inspect))).toBe(true)
    } finally {
      await launchctl(["bootout", target])
      await launchctl(["enable", target])
    }
  }, 20000)
})
