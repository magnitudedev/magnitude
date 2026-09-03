import { Command } from "@commander-js/extra-typings"
import { afterEach, describe, expect, it, vi } from "vitest"
import { parseJsonCommand, requestedJsonCommand } from "./json-command-line"

afterEach(() => {
  process.exitCode = undefined
  vi.restoreAllMocks()
})

describe("JSON command-line parsing", () => {
  it("recognizes JSON only on the three plugin-facing model commands", () => {
    expect(requestedJsonCommand(["models", "status", "--json"])).toBe("models.status")
    expect(requestedJsonCommand(["models", "load", "id", "--json"])).toBe("models.load")
    expect(requestedJsonCommand(["models", "stop", "--json"])).toBe("models.stop")
    expect(requestedJsonCommand(["catalog", "list", "--json"])).toBeUndefined()
    expect(requestedJsonCommand(["models", "--json", "status"])).toBeUndefined()
    expect(requestedJsonCommand(["models", "status"])).toBeUndefined()
    expect(requestedJsonCommand(["models", "status", "--", "--json"])).toBeUndefined()
    expect(requestedJsonCommand(["models", "status", "--json", "--", "--help"])).toBe("models.status")
  })

  it("leaves help on the established human-readable path", () => {
    expect(requestedJsonCommand(["models", "status", "--json", "--help"])).toBeUndefined()
    expect(requestedJsonCommand(["models", "load", "-h", "--json"])).toBeUndefined()
  })

  it("converts a Commander argument error into exactly one JSON failure", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    const program = new Command().name("magnitude")
    program.command("models").command("load")
      .argument("<model-id>")
      .option("--json")

    await parseJsonCommand(program, ["models", "load", "--json"], "models.load")

    expect(stderr).toHaveBeenCalledOnce()
    expect(JSON.parse(String(stderr.mock.calls[0]![0]))).toEqual({
      schemaVersion: 1,
      command: "models.load",
      ok: false,
      error: { message: "missing required argument 'model-id'" },
    })
    expect(process.exitCode).toBe(1)
  })

  it("converts an unknown option into one JSON failure", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    const program = new Command().name("magnitude")
    program.command("models").command("stop").option("--json")

    await parseJsonCommand(program, ["models", "stop", "--json", "--unknown"], "models.stop")

    expect(stderr).toHaveBeenCalledOnce()
    expect(JSON.parse(String(stderr.mock.calls[0]![0]))).toMatchObject({
      schemaVersion: 1,
      command: "models.stop",
      ok: false,
      error: { message: "unknown option '--unknown'" },
    })
    expect(process.exitCode).toBe(1)
  })
})
