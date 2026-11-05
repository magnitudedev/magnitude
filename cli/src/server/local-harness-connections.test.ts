import { Effect } from "effect"
import { rm } from "node:fs/promises"
import { homedir } from "node:os"
import { afterAll, describe, expect, it, vi } from "vitest"

// The caller's home is fixed before the module under test loads, since the desktop host reads it
// then; loading the module is collection's, never the test's.
const caller = await vi.hoisted(async () => {
  const { mkdtemp } = await import("node:fs/promises")
  const { tmpdir } = await import("node:os")
  const { join } = await import("node:path")
  const previous = { ...process.env }
  const home = await mkdtemp(join(tmpdir(), "magnitude-caller-home-"))
  process.env.HOME = home
  for (const name of ["PI_CODING_AGENT_DIR", "CODEX_HOME", "HERMES_HOME", "OPENCLAW_STATE_DIR", "CLINE_DATA_DIR", "PI_CONFIG_DIR"]) delete process.env[name]
  return { home, previous }
})
const captured = vi.hoisted(() => ({ options: undefined as undefined | { paths: Record<string, unknown>; serviceEndpoint: string } }))
vi.mock("@magnitudedev/harness-connections", async original => {
  const actual = await original<typeof import("@magnitudedev/harness-connections")>()
  return { ...actual, makeHarnessConnectionService: (options: never) => Effect.sync(() => { captured.options = options; return {} }) }
})
import { localHarnessConnections } from "./local-harness-connections"

describe("CLI harness connections", () => {
  afterAll(async () => { process.env = { ...caller.previous }; await rm(caller.home, { recursive: true, force: true }) })

  it("writes into the caller's home and points at the service on this machine, never the service's data", async () => {
    await Effect.runPromise(localHarnessConnections)
    const options = captured.options!
    expect(options.serviceEndpoint).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/)
    const paths = JSON.stringify(options.paths)
    const files = [...paths.matchAll(/"(\/[^"]+)"/g)].map(match => match[1]!)
    expect(files.length).toBeGreaterThan(5)
    for (const file of files) expect(file.startsWith(homedir())).toBe(true)
    expect(paths).not.toContain("/var/lib/magnitude")
  })
})
