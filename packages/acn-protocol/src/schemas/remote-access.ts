import { Schema } from "effect"

/** Paths of the HTTP routes a browser on another device uses to sign in to ACN. */
export const REMOTE_ACCESS_SESSION_PATH = "/auth/session"
export const REMOTE_ACCESS_SIGN_OUT_PATH = "/auth/session/end"

/**
 * Header carrying the local control token. A service running as a system service admits loopback
 * callers to application control only with it; anywhere else loopback alone admits them.
 */
export const LOCAL_CONTROL_HEADER = "x-magnitude-control-token"

/**
 * How ACN admits the caller asking. Local callers (loopback, plus the local control token where the
 * service requires one) never sign in; any other browser signs in with the Network access key.
 */
export const RemoteAccessStatus = Schema.Union(
  Schema.TaggedStruct("Local", {}),
  Schema.TaggedStruct("SignedIn", {}),
  Schema.TaggedStruct("SignInRequired", { keyConfigured: Schema.Boolean }),
)
export type RemoteAccessStatus = typeof RemoteAccessStatus.Type

export const SignInRequest = Schema.Struct({ key: Schema.String })
export type SignInRequest = typeof SignInRequest.Type

export class SignInRefused extends Schema.TaggedError<SignInRefused>()("SignInRefused", {
  reason: Schema.Literal("WrongKey", "NoKey", "NetworkAccessOff", "CrossOrigin"),
}) {}
export class SignInThrottled extends Schema.TaggedError<SignInThrottled>()("SignInThrottled", {
  retryAfterSeconds: Schema.Number,
}) {}
export const SignInFailure = Schema.Union(SignInRefused, SignInThrottled)
export type SignInFailure = typeof SignInFailure.Type
