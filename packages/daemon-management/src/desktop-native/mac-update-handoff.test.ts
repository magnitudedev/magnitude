import { Effect } from "effect"
import { EventEmitter } from "node:events"
import { spawn } from "node:child_process"
import { PassThrough, Writable } from "node:stream"
import { afterEach, describe, expect, it, vi } from "vitest"
import { startMacUpdateHandoff, relaunchMacAfterUpdate, type MacUpdateHandoffRequest } from "./mac-update-handoff"
vi.mock("node:child_process", () => ({ spawn: vi.fn() }))
afterEach(() => vi.clearAllMocks())
const request: MacUpdateHandoffRequest = { stateDirectory: "/test/state", helperDirectory: "/test/state/update-helpers/helper-12345678-1234-1234-1234-123456789abc", bundle: "/test/Magnitude.app", showWindow: false }

describe("Mac update relaunch handoff", () => {
  it.each([false, true])("retires an uncommitted helper but retains a committed lifetime pipe (commit=%s)", async commit => {
    const stdout = new PassThrough()
    const stdin = new Writable({ write(_chunk, _encoding, done) { done(); queueMicrotask(() => stdout.write("ready\n")) } })
    const kill = vi.fn()
    const unref = vi.fn()
    vi.mocked(spawn).mockReturnValue(Object.assign(new EventEmitter(), { stdin, stdout, kill, unref }) as unknown as ReturnType<typeof spawn>)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const handoff = yield* startMacUpdateHandoff(request)
      expect(kill).not.toHaveBeenCalled()
      if (commit) yield* handoff.commit
    })))
    expect(kill).toHaveBeenCalledTimes(commit ? 0 : 1)
    expect(stdin.destroyed).toBe(!commit)
    expect(unref).toHaveBeenCalledTimes(commit ? 1 : 0)
    stdin.destroy(); stdout.destroy()
  })
  it.each([false, true])("preserves live relaunch intent (window=%s)", async showWindow => {
    vi.mocked(spawn).mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), { unref: vi.fn() })
      queueMicrotask(() => child.emit("spawn"))
      return child as unknown as ReturnType<typeof spawn>
    })
    await Effect.runPromise(relaunchMacAfterUpdate({ ...request, showWindow }))
    expect(vi.mocked(spawn).mock.calls[0]).toEqual(["/test/Magnitude.app/Contents/MacOS/Magnitude", showWindow ? [] : ["--background"], { detached: true, stdio: "ignore" }])
  })
})
