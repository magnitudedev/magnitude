import { CommandExecutor } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { EventEmitter } from "node:events"
import { spawn } from "node:child_process"
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { PassThrough, Writable } from "node:stream"
import { afterEach, describe, expect, it, vi } from "vitest"
import { completeLinuxUpdateHandoff, LinuxUpdateHandoffRequest, LinuxUpdateResult, startLinuxUpdateHandoff } from "./linux-update-handoff"

vi.mock("node:child_process", () => ({ spawn: vi.fn() }))
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks() })
const request = (directory: string): LinuxUpdateHandoffRequest => ({ stateDirectory: directory, requestPath: join(directory, "application-updates/prepared-123/request.json"), version: "2.0.0", showWindow: false })

describe("Linux update handoff", () => {
  it("keeps the lifetime pipe open after handing off install intent", async () => {
    const chunks: Buffer[] = []
    const acknowledgement = new PassThrough()
    const stdin = new Writable({ write(chunk, _, done) { chunks.push(Buffer.from(chunk)); done(); queueMicrotask(() => acknowledgement.write("ready\n")) } })
    vi.mocked(spawn).mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), { stdin, stdio: [stdin, null, null, acknowledgement], unref: vi.fn(), kill: vi.fn() })
      queueMicrotask(() => child.emit("spawn"))
      return child as unknown as ReturnType<typeof spawn>
    })
    await Effect.runPromise(startLinuxUpdateHandoff(request("/tmp/magnitude-handoff")))
    expect(stdin.writableEnded).toBe(false)
    expect(Schema.decodeUnknownSync(Schema.parseJson(LinuxUpdateHandoffRequest))(Buffer.concat(chunks).toString())).toEqual(request("/tmp/magnitude-handoff"))
    expect(vi.mocked(spawn).mock.calls[0]?.slice(0, 2)).toEqual(["/usr/lib/magnitude-desktop/resources/magnitude", ["_complete-application-update"]])
    stdin.destroy()
  })
  it("rejects staging paths outside its private update directory", () => {
    expect(Schema.decodeUnknownEither(LinuxUpdateHandoffRequest)({ ...request("/tmp/state"), requestPath: "/home/user/request.json" })._tag).toBe("Left")
  })
  it("retires an unready helper before releasing its lifetime channel", async () => {
    const stdin = new Writable({ write(_, __, done) { done() } })
    const acknowledgement = new PassThrough()
    const kill = vi.fn()
    vi.mocked(spawn).mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), { stdin, stdio: [stdin, null, null, acknowledgement], unref: vi.fn(), kill })
      queueMicrotask(() => child.emit("spawn"))
      return child as unknown as ReturnType<typeof spawn>
    })
    await Effect.runPromise(startLinuxUpdateHandoff(request("/tmp/magnitude-handoff")).pipe(Effect.timeoutOption("20 millis")))
    expect(kill).toHaveBeenCalledOnce()
    expect(stdin.destroyed).toBe(true)
    expect(acknowledgement.destroyed).toBe(true)
  })
  it.each([0, 126, 127, 100])("records package/authorization outcome %s and relaunches as the user", async code => {
    vi.stubGlobal("process", { ...process, platform: "linux", getuid: () => 1000 })
    const directory = await mkdtemp(join(tmpdir(), "linux-handoff-"))
    const input = request(directory)
    await mkdir(join(directory, "application-updates/prepared-123"), { recursive: true })
    vi.mocked(spawn).mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), { unref: vi.fn() })
      queueMicrotask(() => child.emit("spawn"))
      return child as unknown as ReturnType<typeof spawn>
    })
    const executor = CommandExecutor.makeExecutor(() => Effect.die("Unexpected native process"))
    try {
      await Effect.runPromise(completeLinuxUpdateHandoff(input).pipe(Effect.provideService(CommandExecutor.CommandExecutor, {
        ...executor, exitCode: command => Effect.sync(() => {
          expect(command._tag).toBe("StandardCommand")
          if (command._tag === "StandardCommand") {
            expect(command.command).toBe("/usr/bin/pkexec")
            expect(command.args).toEqual(["--disable-internal-agent", "/usr/lib/magnitude-desktop/resources/magnitude", "_install-application-update", input.requestPath])
          }
          return CommandExecutor.ExitCode(code)
        }),
      }), Effect.provide(BunContext.layer)))
      const result = Schema.decodeUnknownSync(Schema.parseJson(LinuxUpdateResult))(await readFile(join(directory, "update-result.json"), "utf8"))
      expect(result.error._tag).toBe(code === 0 ? "None" : "Some")
      expect(vi.mocked(spawn).mock.calls[0]?.slice(0, 2)).toEqual(["/usr/bin/magnitude-desktop", ["--background"]])
    } finally { await rm(directory, { recursive: true, force: true }) }
  })
})
