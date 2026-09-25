import { spawn } from "node:child_process"
import { readFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"
import type { Duplex } from "node:stream"
import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { GuardedCommand, guardedCommandLayer } from "./guarded-command"

const helper = fileURLToPath(new URL(`../../dist/native/${process.platform}-${process.arch}/magnitude-command`, import.meta.url))

describe.skipIf(process.platform === "win32")("Protected command lifetime", () => {
  it("returns the command status and both output streams", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const command = yield* GuardedCommand
      return yield* command.run("/bin/sh", ["-c", "printf output; printf error >&2; exit 7"], {})
    }).pipe(Effect.provide(guardedCommandLayer(helper))))
    expect(result).toEqual({ code: 7, stdout: "output", stderr: "error" })
  })

  it.skipIf(process.platform !== "linux")("retires descendants that start a new session when the parent lifetime ends", async () => {
    const child = spawn(helper, ["/usr/bin/python3", "-c", [
      "import os, subprocess, time",
      "nested = subprocess.Popen(['/bin/sleep', '60'], start_new_session=True)",
      "print(str(os.getpid()) + ' ' + str(nested.pid), flush=True)",
      "time.sleep(60)",
    ].join("\n")], { detached: true, stdio: ["ignore", "pipe", "pipe", "pipe", "pipe"] })
    const closed = new Promise<void>((resolve, reject) => { child.once("close", () => resolve()); child.once("error", reject) })
    const lifetime = child.stdio[3] as Duplex
    const pids: number[] = []
    try {
      const output = await new Promise<string>((resolve, reject) => {
        let buffer = ""
        child.stdout!.on("data", chunk => { buffer += chunk; if (buffer.includes("\n")) resolve(buffer) })
        child.once("error", reject)
        child.once("exit", () => reject(new Error("Command exited before reporting descendants")))
      })
      pids.push(...output.trim().split(" ").map(Number))
      expect(pids).toHaveLength(2)
      expect(pids.every(pid => Number.isSafeInteger(pid) && pid > 1)).toBe(true)
      const nested = (await readFile(`/proc/${pids[1]}/stat`, "utf8")).split(") ")[1]!.split(" ")
      expect(Number(nested[3])).toBe(pids[1])
      lifetime.destroy()
      await closed
      for (const pid of pids) await expect(readFile(`/proc/${pid}/stat`)).rejects.toMatchObject({ code: "ENOENT" })
    } finally {
      lifetime.destroy()
      for (const pid of pids) { try { process.kill(pid, "SIGKILL") } catch {} }
    }
  }, 10_000)
})
