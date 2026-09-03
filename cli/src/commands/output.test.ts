import { Data, Effect, Schema } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { renderFields, renderTable, runCommand } from "./output"

class ExpectedFailure extends Data.TaggedError("ExpectedFailure")<{
  readonly message: string
}> {}

afterEach(() => {
  process.exitCode = undefined
  vi.restoreAllMocks()
})

describe("command output", () => {
  it("renders a successful result to stdout", async () => {
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    await runCommand({
      effect: Effect.succeed({ value: "ok" }),
      render: ({ value }) => `${value}\n`,
    })
    expect(stdout).toHaveBeenCalledOnce()
    expect(stdout).toHaveBeenCalledWith("ok\n")
    expect(stderr).not.toHaveBeenCalled()
  })

  it("writes an actionable failure only to stderr", async () => {
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    await runCommand({
      effect: Effect.fail(new ExpectedFailure({ message: "The command is not valid now" })),
      render: () => "unused\n",
    })
    expect(stdout).not.toHaveBeenCalled()
    expect(stderr).toHaveBeenCalledOnce()
    expect(stderr).toHaveBeenCalledWith("The command is not valid now\n")
    expect(process.exitCode).toBe(1)
  })

  it("prefers an actionable reason to an inherited generic message", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    const failure = Object.assign(Object.create({ message: "Generic failure" }) as object, {
      reason: "Magnitude service is not responding",
    })
    await runCommand({
      effect: Effect.fail(failure),
      render: () => "unused\n",
    })
    expect(stderr).toHaveBeenCalledWith("Magnitude service is not responding\n")
  })

  it("writes one schema-versioned JSON success document only to stdout", async () => {
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    await runCommand({
      effect: Effect.succeed({ value: "ok" }),
      render: () => "human output\n",
      json: {
        command: "models.stop",
        schema: Schema.Struct({ value: Schema.String }),
        data: (result) => result,
      },
    })

    expect(stdout).toHaveBeenCalledOnce()
    const output = String(stdout.mock.calls[0]![0])
    expect(output).toBe('{"schemaVersion":1,"command":"models.stop","ok":true,"data":{"value":"ok"}}\n')
    expect(JSON.parse(output)).toEqual({
      schemaVersion: 1,
      command: "models.stop",
      ok: true,
      data: { value: "ok" },
    })
    expect(stderr).not.toHaveBeenCalled()
  })

  it("writes one JSON failure document only to stderr and preserves the nonzero exit", async () => {
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    await runCommand({
      effect: Effect.fail(new ExpectedFailure({ message: "Cannot stop the model" })),
      render: () => "unused\n",
      json: {
        command: "models.stop",
        schema: Schema.Struct({ outcome: Schema.Literal("stopped") }),
        data: () => ({ outcome: "stopped" as const }),
      },
    })

    expect(stdout).not.toHaveBeenCalled()
    expect(stderr).toHaveBeenCalledOnce()
    expect(JSON.parse(String(stderr.mock.calls[0]![0]))).toEqual({
      schemaVersion: 1,
      command: "models.stop",
      ok: false,
      error: { message: "Cannot stop the model" },
    })
    expect(process.exitCode).toBe(1)
  })

  it("turns an invalid success projection into a structured JSON command failure", async () => {
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    await runCommand({
      effect: Effect.succeed(""),
      render: () => "unused\n",
      json: {
        command: "models.load",
        schema: Schema.Struct({ value: Schema.String.pipe(Schema.minLength(1)) }),
        data: (value) => ({ value }),
      },
    })

    expect(stdout).not.toHaveBeenCalled()
    expect(stderr).toHaveBeenCalledOnce()
    const failure = JSON.parse(String(stderr.mock.calls[0]![0]))
    expect(failure).toMatchObject({
      schemaVersion: 1,
      command: "models.load",
      ok: false,
    })
    expect(failure.error.message).toContain("Expected a string at least 1 character(s) long")
    expect(process.exitCode).toBe(1)
  })

  it("renders comparable rows as a borderless table", () => {
    expect(renderTable([{ name: "Codex", status: "Connected" }], [
      { heading: "NAME", value: ({ name }) => name },
      { heading: "STATUS", value: ({ status }) => status },
    ], 80)).toBe("NAME   STATUS\nCodex  Connected\n")
  })

  it("falls back to labeled blocks instead of truncating narrow output", () => {
    expect(renderTable([{ name: "Long model", id: "exact:model:id" }], [
      { heading: "MODEL", value: ({ name }) => name },
      { heading: "MODEL ID", value: ({ id }) => id },
    ], 10)).toContain("MODEL ID  exact:model:id")
  })

  it("aligns labeled detail fields", () => {
    expect(renderFields([["ID", "model"], ["Runtime", "Ready"]])).toBe(
      "  ID       model\n  Runtime  Ready",
    )
  })
})
