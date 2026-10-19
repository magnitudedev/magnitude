import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { atMostOnce, replaySafe } from "../transport/recovery"
import { HuggingFaceSignInFailed } from "../errors"
import {
  HuggingFaceAccountSchema,
  HuggingFaceBrowserSignInSchema,
  HuggingFaceDeviceSignInSchema,
  HuggingFaceSignInAttemptIdSchema,
  HuggingFaceSignInOutcomeSchema,
} from "../schemas/hugging-face"

const GetHuggingFaceAccount = Rpc.make("GetHuggingFaceAccount", {
  payload: Schema.Struct({}),
  success: HuggingFaceAccountSchema,
}).pipe(replaySafe)

/** Starts a browser sign-in and returns the page to open; ACN receives the redirect on this computer. It replaces any sign-in in progress. */
const StartHuggingFaceSignIn = Rpc.make("StartHuggingFaceSignIn", {
  payload: Schema.Struct({}),
  success: HuggingFaceBrowserSignInSchema,
  error: HuggingFaceSignInFailed,
}).pipe(atMostOnce)

/** Starts a device-code sign-in for a terminal or another device; ACN polls until it completes, expires or is cancelled. */
const StartHuggingFaceDeviceSignIn = Rpc.make("StartHuggingFaceDeviceSignIn", {
  payload: Schema.Struct({}),
  success: HuggingFaceDeviceSignInSchema,
  error: HuggingFaceSignInFailed,
}).pipe(atMostOnce)

/** How the attempt a start returned ended, independently of the account a later or earlier attempt left. */
const GetHuggingFaceSignInAttempt = Rpc.make("GetHuggingFaceSignInAttempt", {
  payload: Schema.Struct({ attemptId: HuggingFaceSignInAttemptIdSchema }),
  success: HuggingFaceSignInOutcomeSchema,
}).pipe(replaySafe)

/** Ends the sign-in in progress, or only the named attempt if it is still the one in progress. */
const CancelHuggingFaceSignIn = Rpc.make("CancelHuggingFaceSignIn", {
  payload: Schema.Struct({
    attemptId: Schema.optionalWith(HuggingFaceSignInAttemptIdSchema, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({}),
}).pipe(replaySafe)

const SignOutOfHuggingFace = Rpc.make("SignOutOfHuggingFace", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: HuggingFaceSignInFailed,
}).pipe(replaySafe)

export const HuggingFace = {
  getHuggingFaceAccount: GetHuggingFaceAccount,
  startHuggingFaceSignIn: StartHuggingFaceSignIn,
  startHuggingFaceDeviceSignIn: StartHuggingFaceDeviceSignIn,
  getHuggingFaceSignInAttempt: GetHuggingFaceSignInAttempt,
  cancelHuggingFaceSignIn: CancelHuggingFaceSignIn,
  signOutOfHuggingFace: SignOutOfHuggingFace,
}
