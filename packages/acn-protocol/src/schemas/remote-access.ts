import { Schema } from "effect"

/** Paths of the HTTP routes a browser on another device uses to sign in to ACN. */
export const REMOTE_ACCESS_SESSION_PATH = "/auth/session"
export const REMOTE_ACCESS_SIGN_OUT_PATH = "/auth/session/end"

/**
 * How ACN admits the caller asking. Callers on the server's own machine never sign in; a browser on
 * another device signs in with the Network access key.
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
