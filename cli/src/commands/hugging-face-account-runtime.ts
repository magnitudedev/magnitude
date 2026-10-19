import { huggingFaceAccountSummary, huggingFaceSignedOutMessage } from "@magnitudedev/client-common"
import {
  HuggingFaceAccountSchema,
  HuggingFaceDeviceSignInSchema,
  type HuggingFaceAccount,
  type HuggingFaceDeviceSignIn,
  type HuggingFaceSignedOutReason,
  type HuggingFaceSignInAttemptId,
  type HuggingFaceSignInOutcome,
} from "@magnitudedev/sdk"
import { Cause, Data, Duration, Effect, Option, Schema } from "effect"
import { HUGGING_FACE_LOGIN_COMMAND } from "./hugging-face-access"
import { withClient, type CliModelsClient } from "./inference-client"
import { asJson, commandEffect, jsonDocument, renderCommandError, type JsonOption } from "./output"

export const execute = (command: Effect.Effect<void>): Promise<void> => Effect.runPromise(command)

/** A sign-in that ended without an account: declined, expired, failed, or cancelled. */
export class HuggingFaceSignInIncomplete extends Data.TaggedError("HuggingFaceSignInIncomplete")<{
  readonly message: string
  readonly code: string
}> {}

const signInCancelled = () => new HuggingFaceSignInIncomplete({
  message: "Sign-in cancelled.",
  code: "hugging_face_sign_in_cancelled",
})

const STOPPED = signInCancelled()

const signedOutCodes: Record<HuggingFaceSignedOutReason, string> = {
  Denied: "hugging_face_sign_in_denied",
  Expired: "hugging_face_sign_in_expired",
  Failed: "hugging_face_sign_in_failed",
  SessionExpired: "hugging_face_session_expired",
}

const signInIncomplete = (reason: Option.Option<HuggingFaceSignedOutReason>): HuggingFaceSignInIncomplete =>
  Option.match(reason, {
    onNone: signInCancelled,
    onSome: (value) => new HuggingFaceSignInIncomplete({ message: huggingFaceSignedOutMessage(value), code: signedOutCodes[value] }),
  })

const SIGN_IN_HINT = `Sign in: ${HUGGING_FACE_LOGIN_COMMAND}`

export const renderAccount = (account: HuggingFaceAccount): string => {
  switch (account._tag) {
    case "SignedIn": return `Signed in to Hugging Face as ${account.username}\n`
    case "SigningIn": return `Signing in to Hugging Face. ${huggingFaceAccountSummary(account)}\n`
    case "SignedOut": return Option.match(account.reason, {
      onNone: () => `Not signed in to Hugging Face. Sign in for gated models, private repositories and higher download limits.\n${SIGN_IN_HINT}\n`,
      onSome: (reason) => `${huggingFaceSignedOutMessage(reason)}\n${SIGN_IN_HINT}\n`,
    })
  }
}

export const accountJson = jsonDocument(HuggingFaceAccountSchema, (account: HuggingFaceAccount) => account)

export const showHuggingFaceStatus = (options: JsonOption = {}) => commandEffect({
  effect: withClient((client) => client.huggingFace.getHuggingFaceAccount({})),
  render: renderAccount,
  json: asJson(options, accountJson),
})

export const signOutOfHuggingFace = (options: JsonOption = {}) => commandEffect({
  effect: withClient((client) => client.huggingFace.signOutOfHuggingFace({}).pipe(
    Effect.zipRight(client.huggingFace.getHuggingFaceAccount({})),
  )),
  render: () => "Signed out of Hugging Face.\n",
  json: asJson(options, accountJson),
})

/** "15 minutes", "1 minute", or "45 seconds". */
export const formatValidity = (seconds: number): string => {
  if (seconds < 60) return `${seconds} second${seconds === 1 ? "" : "s"}`
  const minutes = Math.round(seconds / 60)
  return `${minutes} minute${minutes === 1 ? "" : "s"}`
}

export const renderDeviceSignIn = ({ verificationUri, userCode, expiresIn }: HuggingFaceDeviceSignIn): string => [
  "To sign in to Hugging Face, open this page on any device:",
  `  ${verificationUri}`,
  "and enter this code:",
  `  ${userCode}`,
  `The code is valid for ${formatValidity(expiresIn)}. Waiting for you to finish… (Ctrl+C cancels)`,
  "",
].join("\n")

/** Polls this command's own attempt until it ends, so another client's sign-in or an earlier account never answers for it. */
export const awaitSignIn = (
  client: CliModelsClient,
  attemptId: HuggingFaceSignInAttemptId,
  pollInterval: Duration.DurationInput,
): Effect.Effect<Exclude<HuggingFaceSignInOutcome, { readonly _tag: "Pending" }>, unknown> => Effect.sleep(pollInterval).pipe(
  Effect.zipRight(client.huggingFace.getHuggingFaceSignInAttempt({ attemptId })),
  Effect.flatMap((outcome) => outcome._tag === "Pending" ? awaitSignIn(client, attemptId, pollInterval) : Effect.succeed(outcome)),
)

const attemptIncomplete = (outcome: Exclude<HuggingFaceSignInOutcome, { readonly _tag: "Pending" | "SignedIn" }>): HuggingFaceSignInIncomplete => {
  switch (outcome._tag) {
    case "Failed": return signInIncomplete(Option.some(outcome.reason))
    case "Cancelled": return signInCancelled()
    case "Superseded": return new HuggingFaceSignInIncomplete({
      message: "Another Hugging Face sign-in replaced this one.",
      code: "hugging_face_sign_in_superseded",
    })
    case "Unknown": return new HuggingFaceSignInIncomplete({
      message: "Magnitude restarted during the sign-in. Try again.",
      code: "hugging_face_sign_in_failed",
    })
  }
}

/** Ctrl+C or a termination request while waiting: fails so the sign-in is interrupted and cancelled. */
const stopRequested = Effect.async<never, HuggingFaceSignInIncomplete>((resume) => {
  const stop = () => resume(Effect.fail(STOPPED))
  const signals = ["SIGINT", "SIGTERM", "SIGHUP"] as const
  for (const signal of signals) process.once(signal, stop)
  return Effect.sync(() => { for (const signal of signals) process.removeListener(signal, stop) })
})

const write = (output: string) => Effect.sync(() => { process.stdout.write(output) })

const jsonLine = <A, I>(schema: Schema.Schema<A, I, never>, value: A) =>
  Schema.encode(schema)(value).pipe(Effect.map((document) => `${JSON.stringify(document)}\n`))

/**
 * Device sign-in: ACN starts it and polls Hugging Face itself; this prints where to enter the code,
 * waits for its own attempt to end, and cancels that attempt when interrupted.
 */
export const signInDevice = (
  client: CliModelsClient,
  json: boolean,
  pollInterval: Duration.DurationInput,
): Effect.Effect<Extract<HuggingFaceAccount, { readonly _tag: "SignedIn" }>, unknown> => Effect.gen(function* () {
  const start = yield* client.huggingFace.startHuggingFaceDeviceSignIn({})
  yield* json ? jsonLine(HuggingFaceDeviceSignInSchema, start).pipe(Effect.flatMap(write)) : write(renderDeviceSignIn(start))
  const outcome = yield* awaitSignIn(client, start.attemptId, pollInterval).pipe(
    Effect.onInterrupt(() => client.huggingFace.cancelHuggingFaceSignIn({ attemptId: Option.some(start.attemptId) }).pipe(Effect.ignore)),
  )
  if (outcome._tag !== "SignedIn") return yield* attemptIncomplete(outcome)
  const account = yield* client.huggingFace.getHuggingFaceAccount({})
  return account._tag === "SignedIn" ? account : { _tag: "SignedIn" as const, username: outcome.username, avatarUrl: Option.none<string>() }
})

export const signInToHuggingFace = (
  options: JsonOption = {},
  pollInterval: Duration.DurationInput = Duration.seconds(1),
): Effect.Effect<void> => {
  const json = options.json === true
  return withClient((client) => Effect.raceFirst(signInDevice(client, json, pollInterval), stopRequested)).pipe(
    Effect.flatMap((account) => json ? jsonLine(HuggingFaceAccountSchema, account) : Effect.succeed(renderAccount(account))),
    Effect.flatMap(write),
    Effect.catchAllCause((cause) => Effect.sync(() => {
      const error = Cause.squash(cause)
      process.stderr.write(renderCommandError(error, json))
      process.exitCode = error === STOPPED ? 130 : 1
    })),
  )
}
