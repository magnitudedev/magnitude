import { Data, Effect, Schema } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { jsonDocument, renderFields, renderTable, runCommand } from "./output"

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

  it("reports a decode defect as a message instead of crashing", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    const decode = Effect.sync(() => Schema.decodeUnknownSync(Schema.Struct({ contextLength: Schema.Positive }))({ contextLength: 0 }))
    await expect(runCommand({ effect: decode, render: () => "unused\n" })).resolves.toBeUndefined()
    expect(stderr).toHaveBeenCalledOnce()
    expect(String(stderr.mock.calls[0]![0])).toContain("contextLength")
    expect(String(stderr.mock.calls[0]![0])).not.toContain("FiberFailure")
    expect(process.exitCode).toBe(1)
  })

  it("reports a decode defect as one JSON error document under --json", async () => {
    vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    const decode = Effect.sync(() => Schema.decodeUnknownSync(Schema.Struct({ contextLength: Schema.Positive }))({ contextLength: 0 }))
    await runCommand({
      effect: decode,
      render: () => "unused\n",
      json: jsonDocument(Schema.Unknown, (result) => result),
    })
    const document = JSON.parse(String(stderr.mock.calls[0]![0])) as { error: { message: string; code: string } }
    expect(document.error.code).toBe("parse_error")
    expect(document.error.message).toContain("contextLength")
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
