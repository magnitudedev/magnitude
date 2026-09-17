import { spawn } from "node:child_process"
import { once } from "node:events"
import { createInterface } from "node:readline"
import { fileURLToPath } from "node:url"
import { expect, it } from "vitest"

const addon = fileURLToPath(new URL(`../../dist/native/${process.platform}-${process.arch}/desktop-host.node`, import.meta.url))
const fixture = fileURLToPath(new URL("./fixtures/memory.cjs", import.meta.url))

it("measures real allocations across its tree and excludes unrelated processes", async () => {
  const launch = () => {
    const child = spawn(process.execPath, [fixture, addon], { stdio: ["pipe", "pipe", "inherit"] })
    const lines = createInterface({ input: child.stdout })[Symbol.asyncIterator]()
    return { child, read: async (command = "sample") => {
      child.stdin.write(command + "\n")
      const line = await lines.next()
      const sample = JSON.parse(line.value!)
      expect(sample.error).toBeUndefined()
      return sample as { bytes: number; processCount: number; metric: string }
    } }
  }
  const owner = launch(), unrelated = launch()
  try {
    const before = await owner.read()
    expect(before.processCount).toBe(1)
    expect(before.bytes).toBeGreaterThan(0)
    await unrelated.read("self")
    const afterUnrelated = await owner.read()
    expect(afterUnrelated.processCount).toBe(1)
    expect(Math.abs(afterUnrelated.bytes - before.bytes)).toBeLessThan(32 * 1024 * 1024)
    const allocated = await owner.read("child")
    expect(allocated.processCount).toBe(2)
    expect(allocated.bytes - before.bytes).toBeGreaterThan(160 * 1024 * 1024)
    const stopped = await owner.read("stop")
    expect(stopped.processCount).toBe(1)
    expect(allocated.bytes - stopped.bytes).toBeGreaterThan(160 * 1024 * 1024)
    const self = await owner.read("self")
    expect(self.bytes - stopped.bytes).toBeGreaterThan(160 * 1024 * 1024)
  } finally {
    await Promise.all([owner, unrelated].map(async ({ child }) => {
      const exited = once(child, "exit"); child.stdin.end(); await exited
    }))
  }
}, 20000)
