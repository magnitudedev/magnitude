import { CommandExecutor } from "@effect/platform"
import { Effect, Layer, Schema } from "effect"
import { EventEmitter } from "node:events"
import { spawn } from "node:child_process"
import { PassThrough, Writable } from "node:stream"
import { afterEach, describe, expect, it, vi } from "vitest"
import { UpdateRelease } from "@magnitudedev/release/hosted-update"
import { completeWindowsUpdateHandoff, startWindowsUpdateHandoff, WindowsUpdateHandoffRequest, relaunchWindowsAfterUpdate } from "./windows-update-handoff"

import { PreparedUpdate } from "./prepared-update"
import { BunContext } from "@effect/platform-bun"
import { unixPrivateFilePermissions } from "./private-files"
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"

vi.mock("node:child_process", () => ({ spawn: vi.fn() }))
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks() })
const request: WindowsUpdateHandoffRequest = {
  stateDirectory: "C:\\Users\\tester\\Magnitude",
  helperDirectory: "C:\\Users\\tester\\Magnitude\\update-helpers\\helper-12345678-1234-1234-1234-123456789abc",
  applicationPath: "C:\\Users\\tester\\AppData\\Local\\Programs\\Magnitude\\Magnitude.exe",
  dataDirectory: "C:\\Users\\tester\\.magnitude", continuation: { _tag: "Desktop", showWindow: false },
  release: Schema.decodeUnknownSync(UpdateRelease)({ version: "2.0.0", bytes: 1, sha256: "a".repeat(64), signature: "A".repeat(86) + "==" }),
}
it("leaves caller continuation to the invoking command without launching Desktop", async () => {
  await Effect.runPromise(relaunchWindowsAfterUpdate({ ...request, continuation: { _tag: "Caller" } }))
  expect(spawn).not.toHaveBeenCalled()
})

describe("Windows update handoff", () => {
  it.each([
    { helperDirectory: "C:\\unrelated\\helper-12345678-1234-1234-1234-123456789abc" },
    { helperDirectory: request.helperDirectory + "\\..\\other" },
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
    expect(vi.mocked(spawn).mock.calls[0]?.slice(0, 2)).toEqual([`${request.helperDirectory}\\magnitude.exe`, ["_complete-windows-application-update"]])
    expect(vi.mocked(spawn).mock.calls[0]?.[2]?.cwd).toBe(request.helperDirectory)
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
    const directory = await mkdtemp(join(tmpdir(), "windows-update-result-"))
    const attempt = { ...request, dataDirectory: directory }
    await mkdir(join(directory, "updates"))
    await writeFile(join(directory, "updates", "update.json"), Schema.encodeSync(Schema.parseJson(PreparedUpdate))({ release: request.release, installation: { _tag: "Attempted" } }))
    vi.stubGlobal("process", { ...process, platform: "win32", execPath: `${request.helperDirectory}\\magnitude.exe` })
    const events: string[] = []
    vi.mocked(spawn).mockImplementation(() => {
      events.push("relaunch")
      const child = Object.assign(new EventEmitter(), { unref: vi.fn() })
      queueMicrotask(() => child.emit("spawn"))
      return child as unknown as ReturnType<typeof spawn>
    })
    const executor = CommandExecutor.makeExecutor(() => Effect.die("Unexpected process"))
    await Effect.runPromise(completeWindowsUpdateHandoff(attempt).pipe(
      Effect.provideService(CommandExecutor.CommandExecutor, { ...executor, exitCode: command => Effect.sync(() => {
        expect(command._tag === "StandardCommand" && command.command).toBe(`${directory.replaceAll("/", "\\")}\\updates\\magnitude-setup.exe`)
        expect(command._tag === "StandardCommand" && command.args).toEqual(["/S"])
        events.push("installer exited")
        return CommandExecutor.ExitCode(code)
      }) }),

      Effect.provide(unixPrivateFilePermissions.pipe(Layer.provideMerge(BunContext.layer))),
    ))
    expect(events).toEqual(["installer exited"])
    const saved = Schema.decodeUnknownSync(Schema.parseJson(PreparedUpdate))(await readFile(join(directory, "updates", "update.json"), "utf8"))
    expect(saved.installation._tag).toBe(code === 0 ? "Attempted" : "Failed")
    await rm(directory, { recursive: true, force: true })
    await Effect.runPromise(relaunchWindowsAfterUpdate(request))
    expect(vi.mocked(spawn).mock.calls[0]?.slice(0, 2)).toEqual([request.applicationPath, ["--background"]])
  })
})
