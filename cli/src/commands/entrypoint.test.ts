import { describe, expect, it } from "vitest"

const invoke = async (args: readonly string[]) => {
  const process = Bun.spawn([Bun.which("bun")!, new URL("../index.ts", import.meta.url).pathname, ...args], {
    stdin: "ignore", stdout: "pipe", stderr: "pipe",
  })
  try {
    const [code, stdout, stderr] = await Promise.all([process.exited, new Response(process.stdout).text(), new Response(process.stderr).text()])
    return { code, stdout, stderr }
  } finally {
    process.kill()
  }
}

describe("headless CLI entrypoint", () => {
  it("prints help and exits without a terminal", async () => {
    const result = await invoke([])
    expect(result.code).toBe(0)
    expect(result.stdout).toContain("Usage: magnitude")
    expect(result.stdout).toContain("catalog")
    expect(result.stdout).not.toMatch(/\x1b\[/)
    expect(result.stderr).toBe("")
  })
  it.each(["setup", "--prompt", "--resume", "--system-override", "--atif"])("rejects removed interactive input %s", async (argument) => {
    const result = await invoke([argument])
    expect(result.code).not.toBe(0)
    expect(result.stderr).toContain("error:")
    expect(result.stdout).toBe("")
  })
  it.each(["", "catalog", "models", "connections", "service", "docs", "update"])("supports finite help for %s", async (command) => {
    const result = await invoke([...(command ? [command] : []), "--help"])
    expect(result.code).toBe(0)
    expect(result.stdout).toContain("Usage:")
    expect(result.stderr).toBe("")
  })
  it("prints only the version", async () => {
    const result = await invoke(["--version"])
    expect(result.code).toBe(0)
    expect(result.stdout.trim()).toMatch(/^\d+\.\d+\.\d+/)
    expect(result.stdout.trim().split("\n")).toHaveLength(1)
    expect(result.stderr).toBe("")
  })
})
