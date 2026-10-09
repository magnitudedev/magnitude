import { Effect } from "effect"
import { mkdtemp, rm } from "node:fs/promises"
import { homedir, tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const captured = vi.hoisted(() => ({ options: undefined as undefined | { paths: Record<string, unknown>; serviceEndpoint: string } }))
vi.mock("@magnitudedev/harness-connections", async original => {
  const actual = await original<typeof import("@magnitudedev/harness-connections")>()
  return { ...actual, makeHarnessConnectionService: (options: never) => Effect.sync(() => { captured.options = options; return {} }) }
})

describe("CLI harness connections", () => {
  let home: string
  const previous = { ...process.env }
  beforeEach(async () => {
    home = await mkdtemp(join(tmpdir(), "magnitude-caller-home-"))
    process.env.HOME = home
    for (const name of ["PI_CODING_AGENT_DIR", "CODEX_HOME", "HERMES_HOME", "OPENCLAW_STATE_DIR", "CLINE_DATA_DIR", "PI_CONFIG_DIR"]) delete process.env[name]
  })
  afterEach(async () => { process.env = { ...previous }; await rm(home, { recursive: true, force: true }) })

  it("writes into the caller's home and points at the service on this machine, never the service's data", async () => {
    const { localHarnessConnections } = await import("./local-harness-connections")
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
