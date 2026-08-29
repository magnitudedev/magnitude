import {
  LAUNCH_PROTOCOL_VERSION,
  LAUNCH_PROTOCOL_VERSION_VARIABLE,
  POST_UPDATE_SERVICE_START_EXIT_CODE,
  RELAUNCH_EXIT_CODE,
  type UpdateAction,
} from "@magnitudedev/release"
import { Effect, Option } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { executeUpdate } from "./execute"
import {
  UpdateCommandFailed,
  type CliUpdaterShape,
} from "./updater"

const action: UpdateAction = {
  method: "npm",
  command: "npm",
  args: ["install", "-g", "@magnitudedev/cli@1.2.3"],
}

const updaterWith = (
  runUpdate: CliUpdaterShape["runUpdate"],
): CliUpdaterShape => ({
  packageManager: Option.some("npm"),
  discover: Effect.never,
  updateTarget: Effect.succeed(Option.some("1.2.3")),
  dismissVersion: () => Effect.void,
  runUpdate,
})

const originalProtocolVersion = process.env[LAUNCH_PROTOCOL_VERSION_VARIABLE]

afterEach(() => {
  vi.restoreAllMocks()
  if (originalProtocolVersion === undefined) {
    delete process.env[LAUNCH_PROTOCOL_VERSION_VARIABLE]
  } else {
    process.env[LAUNCH_PROTOCOL_VERSION_VARIABLE] = originalProtocolVersion
  }
})

describe("executeUpdate", () => {
  it("returns success without mutating process exit state", async () => {
    vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    const previousExitCode = process.exitCode

    const exitCode = await Effect.runPromise(executeUpdate(
      updaterWith(() => Effect.void),
      action,
    ))

    expect(exitCode).toBe(0)
    expect(process.exitCode).toBe(previousExitCode)
  })

  it("returns the relaunch code only for a compatible launcher", async () => {
    vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    process.env[LAUNCH_PROTOCOL_VERSION_VARIABLE] = String(LAUNCH_PROTOCOL_VERSION)

    const exitCode = await Effect.runPromise(executeUpdate(
      updaterWith(() => Effect.void),
      action,
      { relaunch: true },
    ))

    expect(exitCode).toBe(RELAUNCH_EXIT_CODE)
  })

  it("asks a compatible launcher to start the service after an explicit update", async () => {
    vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    process.env[LAUNCH_PROTOCOL_VERSION_VARIABLE] = String(LAUNCH_PROTOCOL_VERSION)

    const exitCode = await Effect.runPromise(executeUpdate(
      updaterWith(() => Effect.void),
      action,
    ))

    expect(exitCode).toBe(POST_UPDATE_SERVICE_START_EXIT_CODE)
  })

  it("returns failure without mutating process exit state", async () => {
    vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    const previousExitCode = process.exitCode

    const exitCode = await Effect.runPromise(executeUpdate(
      updaterWith(() => Effect.fail(new UpdateCommandFailed({
        command: "npm",
        reason: "exited with status 7",
      }))),
      action,
    ))

    expect(exitCode).toBe(1)
    expect(process.exitCode).toBe(previousExitCode)
  })
})
