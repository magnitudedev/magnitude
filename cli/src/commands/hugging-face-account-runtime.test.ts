import { Command } from "@commander-js/extra-typings"
import { Duration, Effect, Fiber, Option, Schema, TestClock, TestContext } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  HuggingFaceDeviceSignInSchema,
  HuggingFaceSignInFailed,
  type HuggingFaceAccount,
  type HuggingFaceDeviceSignIn,
  type HuggingFaceSignInAttemptId,
  type HuggingFaceSignInOutcome,
} from "@magnitudedev/sdk"
import { registerHuggingFaceCommand } from "./hugging-face-account"
import {
  formatValidity,
  renderAccount,
  showHuggingFaceStatus,
  signInDevice,
  signInToHuggingFace,
  signOutOfHuggingFace,
} from "./hugging-face-account-runtime"
import { CliModelsConnection, type CliModelsClient } from "./inference-client"

const ATTEMPT = "attempt-1" as HuggingFaceSignInAttemptId
const DEVICE: HuggingFaceDeviceSignIn = {
  attemptId: ATTEMPT,
  verificationUri: "https://huggingface.co/device",
  verificationUriComplete: Option.none(),
  userCode: "WXYZ-1234",
  expiresIn: 900,
  interval: 5,
}

const SIGNING_IN: HuggingFaceAccount = { _tag: "SigningIn", method: "Device" }
const SIGNED_IN: HuggingFaceAccount = { _tag: "SignedIn", username: "alice", avatarUrl: Option.none() }
const signedOut = (reason?: "Denied" | "Expired" | "Failed" | "SessionExpired"): HuggingFaceAccount =>
  ({ _tag: "SignedOut", reason: Option.fromNullable(reason) })

const PENDING: HuggingFaceSignInOutcome = { _tag: "Pending" }
const SUCCEEDED: HuggingFaceSignInOutcome = { _tag: "SignedIn", username: "alice" }
const failed = (reason: "Denied" | "Expired" | "Failed"): HuggingFaceSignInOutcome => ({ _tag: "Failed", reason })

/**
 * A fake service whose attempt reads return `outcomes` in order, repeating the last one, and whose
 * account reads return `account`.
 */
const fakeService = (outcomes: readonly HuggingFaceSignInOutcome[], account: HuggingFaceAccount = SIGNED_IN) => {
  let reads = 0
  const huggingFace = {
    startHuggingFaceDeviceSignIn: vi.fn(() => Effect.succeed(DEVICE)),
    getHuggingFaceSignInAttempt: vi.fn(() => Effect.sync(() => outcomes[Math.min(reads++, outcomes.length - 1)]!)),
    getHuggingFaceAccount: vi.fn(() => Effect.succeed(account)),
    cancelHuggingFaceSignIn: vi.fn(() => Effect.succeed({})),
    signOutOfHuggingFace: vi.fn(() => Effect.succeed({})),
  }
  return { client: { huggingFace } as unknown as CliModelsClient, huggingFace, reads: () => reads }
}

const capture = async (command: Effect.Effect<void>, client: CliModelsClient) => {
  let stdout = ""
  let stderr = ""
  process.exitCode = 0
  const out = vi.spyOn(process.stdout, "write").mockImplementation((chunk) => { stdout += String(chunk); return true })
  const err = vi.spyOn(process.stderr, "write").mockImplementation((chunk) => { stderr += String(chunk); return true })
  await Effect.runPromise(command.pipe(Effect.provideService(CliModelsConnection, { connect: Effect.succeed(client) })))
  out.mockRestore()
  err.mockRestore()
  const exitCode = process.exitCode
  process.exitCode = 0
  return { stdout, stderr, exitCode }
}

const login = (outcomes: readonly HuggingFaceSignInOutcome[], json = false, account: HuggingFaceAccount = SIGNED_IN) => {
  const service = fakeService(outcomes, account)
  return capture(signInToHuggingFace({ json }, Duration.zero), service.client).then((result) => ({ ...result, ...service }))
}

afterEach(() => {
  vi.restoreAllMocks()
  process.exitCode = 0
})

describe("hf command group", () => {
  it("registers login, status and logout, each with --json", () => {
    const program = new Command().name("magnitude")
    registerHuggingFaceCommand(program)
    const group = program.commands.find((command) => command.name() === "hf")!
    expect(group.commands.map((command) => command.name())).toEqual(["login", "status", "logout"])
    for (const command of group.commands) {
      expect(command.description()).not.toBe("")
      expect(command.options.map((option) => option.long)).toEqual(["--json"])
    }
  })
})

describe("hf login", () => {
  it("prints where to enter the code, then who signed in", async () => {
    const { stdout, stderr, exitCode, huggingFace, reads } = await login([PENDING, PENDING, SUCCEEDED])
    expect(stdout).toBe([
      "To sign in to Hugging Face, open this page on any device:",
      "  https://huggingface.co/device",
      "and enter this code:",
      "  WXYZ-1234",
      "The code is valid for 15 minutes. Waiting for you to finish… (Ctrl+C cancels)",
      "Signed in to Hugging Face as alice",
      "",
    ].join("\n"))
    expect(stderr).toBe("")
    expect(exitCode).toBe(0)
    expect(reads()).toBe(3)
    expect(huggingFace.cancelHuggingFaceSignIn).not.toHaveBeenCalled()
  })

  it("prints the sign-in code and the result as one JSON line each under --json", async () => {
    const { stdout, stderr, exitCode } = await login([PENDING, SUCCEEDED], true)
    const lines = stdout.trimEnd().split("\n")
    expect(lines).toHaveLength(2)
    expect(JSON.parse(lines[0]!)).toEqual(Schema.encodeSync(HuggingFaceDeviceSignInSchema)(DEVICE))
    expect(JSON.parse(lines[1]!)).toEqual({ _tag: "SignedIn", username: "alice" })
    expect(stderr).toBe("")
    expect(exitCode).toBe(0)
  })

  it.each([
    ["Expired", "The sign-in wasn’t finished in time. Try again.", "hugging_face_sign_in_expired"],
    ["Denied", "Sign-in was declined on Hugging Face.", "hugging_face_sign_in_denied"],
    ["Failed", "Magnitude couldn’t sign in to Hugging Face. Try again.", "hugging_face_sign_in_failed"],
  ] as const)("fails when the sign-in ends %s", async (reason, message, code) => {
    const text = await login([PENDING, failed(reason)])
    expect(text.stdout).toContain("WXYZ-1234")
    expect(text.stdout).not.toContain("Signed in")
    expect(text.stderr).toBe(`${message}\n`)
    expect(text.exitCode).toBe(1)

    const json = await login([PENDING, failed(reason)], true)
    expect(json.stdout.trimEnd().split("\n")).toHaveLength(1)
    expect(JSON.parse(json.stderr)).toEqual({ error: { message, code } })
    expect(json.exitCode).toBe(1)
  })

  it("fails as cancelled when the sign-in is cancelled elsewhere", async () => {
    const { stderr, exitCode } = await login([PENDING, { _tag: "Cancelled" }])
    expect(stderr).toBe("Sign-in cancelled.\n")
    expect(exitCode).toBe(1)
  })

  it("never reports an earlier account as its own sign-in", async () => {
    const denied = await login([PENDING, failed("Denied")], false, { _tag: "SignedIn", username: "earlier", avatarUrl: Option.none() })
    expect(denied.stdout).not.toContain("earlier")
    expect(denied.stderr).toBe("Sign-in was declined on Hugging Face.\n")
    expect(denied.exitCode).toBe(1)
    const replaced = await login([PENDING, { _tag: "Superseded" }], false, SIGNING_IN)
    expect(replaced.stderr).toBe("Another Hugging Face sign-in replaced this one.\n")
    expect(replaced.exitCode).toBe(1)
    expect(replaced.huggingFace.getHuggingFaceSignInAttempt).toHaveBeenCalledWith({ attemptId: ATTEMPT })
  })

  it("reports a sign-in Hugging Face wouldn't start", async () => {
    const failure = new HuggingFaceSignInFailed({ message: "Magnitude couldn’t reach Hugging Face. Check your internet connection, then try again." })
    const service = fakeService([SUCCEEDED])
    service.huggingFace.startHuggingFaceDeviceSignIn.mockImplementation(() => Effect.fail(failure) as never)
    const { stdout, stderr, exitCode } = await capture(signInToHuggingFace({ json: true }, Duration.zero), service.client)
    expect(stdout).toBe("")
    expect(JSON.parse(stderr)).toEqual({ error: { message: failure.message, code: "hugging_face_sign_in_failed" } })
    expect(exitCode).toBe(1)
  })

  it("polls once a second and cancels the sign-in when interrupted", async () => {
    const service = fakeService([PENDING])
    vi.spyOn(process.stdout, "write").mockImplementation(() => true)
    await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* Effect.fork(signInDevice(service.client, false, Duration.seconds(1)))
      yield* TestClock.adjust(Duration.seconds(3))
      expect(service.reads()).toBe(3)
      yield* Fiber.interrupt(fiber)
    }).pipe(Effect.provide(TestContext.TestContext)))
    expect(service.huggingFace.cancelHuggingFaceSignIn).toHaveBeenCalledExactlyOnceWith({ attemptId: Option.some(ATTEMPT) })
  })

  it("words how long the code is valid", () => {
    expect(formatValidity(900)).toBe("15 minutes")
    expect(formatValidity(60)).toBe("1 minute")
    expect(formatValidity(45)).toBe("45 seconds")
    expect(formatValidity(1)).toBe("1 second")
  })
})

describe("hf status", () => {
  it("names the signed-in account, in text and JSON", async () => {
    const account: HuggingFaceAccount = { _tag: "SignedIn", username: "alice", avatarUrl: Option.some("https://huggingface.co/a.png") }
    const text = await capture(showHuggingFaceStatus(), fakeService([], account).client)
    expect(text.stdout).toBe("Signed in to Hugging Face as alice\n")
    const json = await capture(showHuggingFaceStatus({ json: true }), fakeService([], account).client)
    expect(JSON.parse(json.stdout)).toEqual({ _tag: "SignedIn", username: "alice", avatarUrl: "https://huggingface.co/a.png" })
  })

  it("points a signed-out account at login, with the reason when there is one", () => {
    expect(renderAccount(signedOut())).toBe([
      "Not signed in to Hugging Face. Sign in for gated models, private repositories and higher download limits.",
      "Sign in: magnitude hf login",
      "",
    ].join("\n"))
    expect(renderAccount(signedOut("SessionExpired"))).toBe([
      "Your Hugging Face sign-in expired. Sign in again.",
      "Sign in: magnitude hf login",
      "",
    ].join("\n"))
    expect(renderAccount(SIGNING_IN)).toBe("Signing in to Hugging Face. Waiting for the code to be entered…\n")
  })

  it("encodes a signed-out account with its reason", async () => {
    const { stdout } = await capture(showHuggingFaceStatus({ json: true }), fakeService([], signedOut("Expired")).client)
    expect(JSON.parse(stdout)).toEqual({ _tag: "SignedOut", reason: "Expired" })
  })
})

describe("hf logout", () => {
  it("signs out and reports the signed-out account", async () => {
    const text = fakeService([], signedOut())
    expect((await capture(signOutOfHuggingFace(), text.client)).stdout).toBe("Signed out of Hugging Face.\n")
    expect(text.huggingFace.signOutOfHuggingFace).toHaveBeenCalledOnce()
    const json = fakeService([], signedOut())
    expect(JSON.parse((await capture(signOutOfHuggingFace({ json: true }), json.client)).stdout)).toEqual({ _tag: "SignedOut" })
  })
})
