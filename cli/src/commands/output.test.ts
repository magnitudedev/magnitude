import { Data, Effect, Option, Schema } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { errorDocument, runCommand, setOutputMode } from "./output"

const ResultSchema = Schema.Struct({
  value: Schema.String,
  detail: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})

class ExpectedFailure extends Data.TaggedError("ExpectedFailure")<{
  readonly code: string
  readonly message: string
  readonly retryable: boolean
}> {}

class ReasonFailure extends Schema.TaggedError<ReasonFailure>()(
  "ReasonFailure",
  { reason: Schema.String },
) {}

afterEach(() => {
  setOutputMode(false)
  process.exitCode = undefined
  vi.restoreAllMocks()
})

describe("unified command output", () => {
  it("schema-encodes exactly one JSON success document", async () => {
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    setOutputMode(true)

    await runCommand({
      effect: Effect.succeed({ value: "ok", detail: Option.none<string>() }),
      schema: ResultSchema,
      render: () => "human\n",
    })

    expect(stdout).toHaveBeenCalledOnce()
    expect(stdout).toHaveBeenCalledWith('{"value":"ok"}\n')
    expect(stderr).not.toHaveBeenCalled()
  })

  it("writes one structured JSON failure only to stderr", async () => {
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    setOutputMode(true)

    await runCommand({
      effect: Effect.fail(new ExpectedFailure({
        code: "wrong_state",
        message: "The command is not valid now",
        retryable: false,
      })),
      schema: ResultSchema,
      render: () => "human\n",
    })

    expect(stdout).not.toHaveBeenCalled()
    expect(stderr).toHaveBeenCalledOnce()
    expect(JSON.parse(String(stderr.mock.calls[0]![0]))).toEqual({
      error: {
        code: "wrong_state",
        message: "The command is not valid now",
        retryable: false,
      },
    })
    expect(process.exitCode).toBe(1)
  })

  it("renders human output from the same decoded result", async () => {
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true)

    await runCommand({
      effect: Effect.succeed({ value: "ok", detail: Option.some("present") }),
      schema: ResultSchema,
      render: ({ value, detail }) => `${value}:${Option.getOrElse(detail, () => "none")}\n`,
    })

    expect(stdout).toHaveBeenCalledWith("ok:present\n")
  })

  it("uses the actionable message instead of a transport reason category", () => {
    expect(errorDocument({
      reason: "Unknown",
      message: "Magnitude service failed to start in time",
    })).toEqual({
      error: {
        code: "command_failed",
        message: "Magnitude service failed to start in time",
        retryable: false,
      },
    })
  })

  it("uses an Effect tagged error's reason instead of its inherited message", () => {
    expect(errorDocument(new ReasonFailure({ reason: "Magnitude service is not responding" }))).toEqual({
      error: {
        code: "command_failed",
        message: "Magnitude service is not responding",
        retryable: false,
      },
    })
  })
})
