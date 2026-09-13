import { CommandExecutor, FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { EventEmitter } from "node:events"
import { spawn } from "node:child_process"
import { PassThrough, Writable } from "node:stream"
import { afterEach, describe, expect, it, vi } from "vitest"
import { SignedUpdateManifest } from "@magnitudedev/release/hosted-update"
import { completeWindowsUpdateHandoff, startWindowsUpdateHandoff, WindowsUpdateHandoffRequest, WindowsUpdateResult } from "./windows-update-handoff"

vi.mock("node:child_process", () => ({ spawn: vi.fn() }))
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks() })
const request: WindowsUpdateHandoffRequest = {
  stateDirectory: "C:\\Users\\tester\\Magnitude",
  preparedDirectory: "C:\\Users\\tester\\Magnitude\\application-updates\\prepared-12345678-1234-1234-1234-123456789abc",
  applicationPath: "C:\\Users\\tester\\AppData\\Local\\Programs\\Magnitude\\Magnitude.exe",
  version: "2.0.0", showWindow: false,
  envelope: Schema.decodeUnknownSync(SignedUpdateManifest)({ keyId: "fixture", payload: "", signature: "" }),
}
describe("Windows update handoff", () => {
  it.each([
    { preparedDirectory: "C:\\unrelated\\prepared-12345678-1234-1234-1234-123456789abc" },
    { preparedDirectory: request.preparedDirectory + "\\..\\other" },
    { applicationPath: "\\\\server\\share\\Magnitude.exe" },
    { applicationPath: "C:\\Windows\\cmd.exe" },
  ])("rejects an unrelated or redirected handoff path", change => {
    expect(Schema.decodeUnknownEither(WindowsUpdateHandoffRequest)({ ...request, ...change })._tag).toBe("Left")
  })
  it("waits for helper readiness while keeping the owner's lifetime pipe open", async () => {
    const stdout = new PassThrough()
    const chunks: Buffer[] = []
    const stdin = new Writable({ write(chunk, _, done) { chunks.push(Buffer.from(chunk)); done(); queueMicrotask(() => stdout.write("ready\n")) } })
    vi.mocked(spawn).mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), { stdin, stdout, kill: vi.fn(), unref: vi.fn() })
      queueMicrotask(() => child.emit("spawn"))
      return child as unknown as ReturnType<typeof spawn>
    })
    await Effect.runPromise(startWindowsUpdateHandoff(request))
    expect(stdin.writableEnded).toBe(false)
    expect(Schema.decodeUnknownSync(Schema.parseJson(WindowsUpdateHandoffRequest))(Buffer.concat(chunks).toString())).toEqual(request)
    expect(vi.mocked(spawn).mock.calls[0]?.slice(0, 2)).toEqual([`${request.preparedDirectory}\\magnitude-update.exe`, ["_complete-windows-application-update"]])
    expect(vi.mocked(spawn).mock.calls[0]?.[2]?.cwd).toBe(request.preparedDirectory)
    stdin.destroy(); stdout.destroy()
  })
  it("retires an unready helper on cancellation", async () => {
    const stdin = new Writable({ write(_, __, done) { done() } })
    const stdout = new PassThrough()
    const kill = vi.fn()
    vi.mocked(spawn).mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), { stdin, stdout, kill, unref: vi.fn() })
      queueMicrotask(() => child.emit("spawn"))
      return child as unknown as ReturnType<typeof spawn>
    })
    await Effect.runPromise(startWindowsUpdateHandoff(request).pipe(Effect.timeoutOption("20 millis")))
    expect(kill).toHaveBeenCalledOnce()
    expect(stdin.destroyed).toBe(true)
    expect(stdout.destroyed).toBe(true)
  })
  it.each([0, 1])("records installer exit %s before relaunch and leaves helper cleanup to the new desktop", async code => {
    vi.stubGlobal("process", { ...process, platform: "win32", execPath: `${request.preparedDirectory}\\magnitude-update.exe` })
    const events: string[] = []
    let record = ""
    vi.mocked(spawn).mockImplementation(() => {
      events.push("relaunch")
      const child = Object.assign(new EventEmitter(), { unref: vi.fn() })
      queueMicrotask(() => child.emit("spawn"))
      return child as unknown as ReturnType<typeof spawn>
    })
    const executor = CommandExecutor.makeExecutor(() => Effect.die("Unexpected process"))
    await Effect.runPromise(completeWindowsUpdateHandoff(request).pipe(
      Effect.provideService(CommandExecutor.CommandExecutor, { ...executor, exitCode: command => Effect.sync(() => {
        expect(command._tag === "StandardCommand" && command.command).toBe(`${request.preparedDirectory}\\magnitude-setup.exe`)
        expect(command._tag === "StandardCommand" && command.args).toEqual(["/S"])
        events.push("installer exited")
        return CommandExecutor.ExitCode(code)
      }) }),
      Effect.provideService(FileSystem.FileSystem, FileSystem.makeNoop({
        writeFileString: (_, contents) => Effect.sync(() => { record = contents; events.push("recorded") }),
        rename: () => Effect.sync(() => { events.push("published") }),
      })),
    ))
    expect(events).toEqual(["installer exited", "recorded", "published", "relaunch"])
    expect(Schema.decodeUnknownSync(Schema.parseJson(WindowsUpdateResult))(record).error._tag).toBe(code === 0 ? "None" : "Some")
    expect(vi.mocked(spawn).mock.calls[0]?.slice(0, 2)).toEqual([request.applicationPath, ["--background"]])
  })
})
