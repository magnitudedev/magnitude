import { createServer, type Server } from "node:net"
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { LegacyStartupCommands } from "./legacy-startup-command"
import { makeLinuxLegacyStartup } from "./legacy-startup-linux"

const source = `[Unit]
Description=Magnitude local inference service
After=network.target

[Service]
Type=simple
ExecStart="/fixture with spaces/magnitude-service" "serve"
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
`
let home: string
let runtime: string
let path: string
let socket: Server | undefined
beforeEach(async () => {
  home = await mkdtemp("/tmp/magnitude-linux-")
  runtime = join(home, "runtime")
  path = join(home, ".config/systemd/user/magnitude.service")
  await mkdir(join(home, ".config/systemd/user"), { recursive: true })
  await mkdir(join(runtime, "systemd"), { recursive: true })
})
afterEach(async () => {
  if (socket) await new Promise<void>((resolve, reject) => socket!.close(error => error ? reject(error) : resolve()))
  socket = undefined
  await rm(home, { recursive: true, force: true })
})
const run = Effect.runPromise
const fixture = async (options: { absent?: boolean; noManager?: boolean; enabled?: string; stopFailure?: boolean; reloadFailure?: boolean } = {}) => {
  if (!options.absent) await writeFile(path, source)
  if (!options.noManager) {
    socket = createServer()
    await new Promise<void>((resolve, reject) => { socket!.once("error", reject); socket!.listen(join(runtime, "systemd/private"), resolve) })
  }
  const state: Record<string, string> = {
    LoadState: options.absent ? "not-found" : "loaded", ActiveState: options.absent ? "inactive" : "active",
    FragmentPath: options.absent ? "" : path, UnitFileState: options.enabled ?? "enabled", MainPID: options.absent ? "0" : "12345",
    DropInPaths: "", Transient: "no", NeedDaemonReload: "no",
  }
  const calls: string[][] = []
  let failReload = options.reloadFailure ?? false
  const commands = LegacyStartupCommands.of({ run: (executable, args) => Effect.gen(function* () {
    expect(executable).toBe("systemctl")
    expect(args.slice(0, 2)).toEqual(["--user", "--no-pager"])
    calls.push([...args.slice(2)])
    const command = args[2]
    if (command === "show") return { code: 0, stdout: Object.entries(state).map(([key, value]) => `${key}=${value}`).join("\n"), stderr: "" }
    if (command === "disable") state.UnitFileState = "disabled"
    else if (command === "stop") {
      if (options.stopFailure) return { code: 1, stdout: "", stderr: "stop failed" }
      state.MainPID = "0"; state.ActiveState = "inactive"
    } else if (command === "daemon-reload") {
      if (failReload) return { code: 1, stdout: "", stderr: "reload failed" }
      const exists = yield* Effect.promise(() => readFile(path).then(() => true).catch(() => false))
      if (!exists) { state.LoadState = "not-found"; state.FragmentPath = "" }
    } else throw new Error(`Unexpected command ${command}`)
    return { code: 0, stdout: "", stderr: "" }
  }) })
  const adapter = await run(makeLinuxLegacyStartup({ home, runtimeDirectory: runtime }).pipe(Effect.provideService(LegacyStartupCommands, commands)))
  return { adapter, state, calls, repairReload: () => { failReload = false } }
}
const mutations = (calls: string[][]) => calls.filter(call => call[0] !== "show").map(call => call.join(" "))

describe.skipIf(process.platform === "win32")("legacy Linux user service", () => {
  it.each(["enabled", "disabled", "enabled-runtime"])("preserves persistent login preference for %s", async enabled => {
    const f = await fixture({ enabled })
    const value = Option.getOrThrow(await run(f.adapter.inspect))
    expect(value.enabled).toBe(enabled === "enabled")
    expect(Option.getOrThrow(value.runningPid)).toBe(12345)
    expect(mutations(f.calls)).toEqual([])
  })
  it("disables, stops, removes the verified source, reloads, and tolerates replay", async () => {
    const f = await fixture()
    const snapshot = Option.getOrThrow(await run(f.adapter.inspect))
    await run(f.adapter.unregister(snapshot))
    expect(mutations(f.calls)).toEqual(["disable magnitude.service", "stop magnitude.service", "daemon-reload"])
    await expect(readFile(path)).rejects.toMatchObject({ code: "ENOENT" })
    await run(f.adapter.unregister(snapshot))
    expect(Option.isNone(await run(f.adapter.inspect))).toBe(true)
  })
  it("removes runtime enablement as well as persistent enablement", async () => {
    const f = await fixture({ enabled: "enabled-runtime" })
    await run(f.adapter.unregister(Option.getOrThrow(await run(f.adapter.inspect))))
    expect(mutations(f.calls).slice(0, 2)).toEqual(["disable --runtime magnitude.service", "disable magnitude.service"])
  })
  it("treats absent source and absent user manager as no legacy registration", async () => {
    const f = await fixture({ absent: true, noManager: true })
    expect(Option.isNone(await run(f.adapter.inspect))).toBe(true)
    expect(f.calls).toEqual([])
  })
  it("does not infer absence when the unit exists but its manager is unavailable", async () => {
    const f = await fixture({ noManager: true })
    await expect(run(f.adapter.inspect)).rejects.toThrow("manager is unavailable")
    expect(f.calls).toEqual([])
  })
  it.each([
    ["FragmentPath", "/another/service"], ["DropInPaths", "/override.conf"], ["NeedDaemonReload", "yes"],
    ["Transient", "yes"], ["UnitFileState", "masked"], ["LoadState", "error"],
  ])("refuses unverified %s", async (key, value) => {
    const f = await fixture()
    f.state[key] = value
    await expect(run(f.adapter.inspect)).rejects.toThrow()
    expect(mutations(f.calls)).toEqual([])
  })
  it("rejects custom hooks and changed source before any manager mutation", async () => {
    const f = await fixture()
    const snapshot = Option.getOrThrow(await run(f.adapter.inspect))
    await writeFile(path, source.replace("Type=simple", "Type=simple\nExecStop=/custom/hook"))
    await expect(run(f.adapter.inspect)).rejects.toThrow("generated Magnitude service")
    await expect(run(f.adapter.unregister(snapshot))).rejects.toThrow("changed after")
    expect(mutations(f.calls)).toEqual([])
  })
  it("refuses a replacement main process", async () => {
    const f = await fixture()
    const snapshot = Option.getOrThrow(await run(f.adapter.inspect))
    f.state.MainPID = "12346"
    await expect(run(f.adapter.unregister(snapshot))).rejects.toThrow("different process")
    expect(mutations(f.calls)).toEqual([])
  })
  it("keeps the source after failed stop", async () => {
    const f = await fixture({ stopFailure: true })
    await expect(run(f.adapter.unregister(Option.getOrThrow(await run(f.adapter.inspect))))).rejects.toThrow("stop failed")
    expect(await readFile(path, "utf8")).toBe(source)
  })
  it("resumes manager reload after removal without restoring the old file", async () => {
    const f = await fixture({ reloadFailure: true })
    const snapshot = Option.getOrThrow(await run(f.adapter.inspect))
    await expect(run(f.adapter.unregister(snapshot))).rejects.toThrow("reload failed")
    await expect(readFile(path)).rejects.toMatchObject({ code: "ENOENT" })
    f.repairReload()
    await run(f.adapter.unregister(snapshot))
    expect(mutations(f.calls).filter(call => call.startsWith("stop"))).toHaveLength(1)
  })
  it("rejects a source symlink", async () => {
    const f = await fixture({ absent: true })
    const target = join(home, "target")
    await writeFile(target, source)
    await symlink(target, path)
    await expect(run(f.adapter.inspect)).rejects.toThrow("regular file")
    expect(mutations(f.calls)).toEqual([])
  })
})
