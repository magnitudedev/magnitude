import { Effect, Schema } from "effect"
import { spawn } from "node:child_process"
import { once } from "node:events"
import { expect, test } from "vitest"
import { ApplicationIdentity } from "../src/application-identity"
import { ProcessExecutorLive } from "../src/process"
import { captureRemovalProcesses, verifyRemovalProcesses, RemovalProcesses } from "../src/suites/uninstall-processes"

test.skipIf(!["darwin", "linux"].includes(process.platform))("native removal captures descendants, rejects survivors and distinguishes reused process IDs", async () => {
  const parent = spawn("python3", ["-u", "-c", `import json,subprocess,time
children=[subprocess.Popen(['python3','-c','import time; time.sleep(120)']) for _ in range(2)]
print(json.dumps([child.pid for child in children]),flush=True)
time.sleep(120)`], { detached: true, stdio: ["ignore", "pipe", "pipe"] })
  const closed = once(parent, "close")
  const exited = once(parent, "exit")
  try {
    const data = await Promise.race([
      once(parent.stdout!, "data"),
      closed.then(() => { throw new Error("Native process fixture exited before readiness") }),
    ])
    const children = Schema.decodeUnknownSync(Schema.parseJson(Schema.Array(Schema.Int)))(data[0].toString())
    const owner = Schema.decodeUnknownSync(ApplicationIdentity)({ applicationPid: parent.pid, servicePid: children[0], serviceInstance: "owned-fixture" })
    const captured = await Effect.runPromise(captureRemovalProcesses(owner).pipe(Effect.provide(ProcessExecutorLive)))
    expect(captured.map(item => item.pid).sort()).toEqual([parent.pid!, ...children].sort())
    expect(captured.every(item => item.executable.startsWith("/") && item.birth.length > 0)).toBe(true)
    const live = await Effect.runPromise(verifyRemovalProcesses(captured).pipe(Effect.either, Effect.provide(ProcessExecutorLive)))
    expect(live._tag).toBe("Left")
    if (live._tag === "Left") expect(live.left._tag).toBe("AssertionFailure")
    const reused = Schema.decodeUnknownSync(RemovalProcesses)(captured.map(item => ({ ...item, birth: "different-creation-time" })))
    await Effect.runPromise(verifyRemovalProcesses(reused).pipe(Effect.provide(ProcessExecutorLive)))
    process.kill(parent.pid!, "SIGTERM")
    process.kill(children[0]!, "SIGTERM")
    await exited
    const orphan = await Effect.runPromise(verifyRemovalProcesses(captured).pipe(Effect.either, Effect.provide(ProcessExecutorLive)))
    expect(orphan._tag).toBe("Left")
    if (orphan._tag === "Left") expect(orphan.left.message).toContain(String(children[1]))
    process.kill(-parent.pid!, "SIGTERM")
    await closed
    await Effect.runPromise(verifyRemovalProcesses(captured).pipe(Effect.provide(ProcessExecutorLive)))
  } finally {
    try { process.kill(-parent.pid!, "SIGKILL") }
    catch (error) { if (!(error instanceof Error && "code" in error && error.code === "ESRCH")) throw error }
    await closed
  }
}, 30_000)
