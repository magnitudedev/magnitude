import { HttpClient, HttpClientRequest } from "@effect/platform"
import { Clock, Config, Effect, Option, Redacted, Schema } from "effect"
import { decodeJwt } from "jose"
import { entraClientToken } from "./entra-token"
import { EntraApplicationId, EntraTenantId } from "./entra-auth"
import { LabApiError } from "./client"

const TokenResponse = Schema.Struct({ value: Schema.NonEmptyString.pipe(Schema.maxLength(32_768)) })
const Lifetime = Schema.Struct({ exp: Schema.Number })
/** Capture the transport once; the returned effect renews independently of lab requests. */
export const githubClientToken = (requestUrl: string, requestToken: Redacted.Redacted<string>, audience: string) => Effect.gen(function* () {
  const url = yield* Effect.try({ try: () => new URL(requestUrl), catch: () => new LabApiError({ status: 0, message: "Invalid GitHub token URL" }) })
  if (url.protocol !== "https:" || url.username || url.password || url.hash || !url.hostname.endsWith(".actions.githubusercontent.com")) {
    return yield* new LabApiError({ status: 0, message: "GitHub token URL must use its HTTPS Actions host" })
  }
  url.searchParams.set("audience", audience)
  const http = yield* HttpClient.HttpClient
  const acquire = Effect.gen(function* () {
    const response = yield* http.execute(HttpClientRequest.get(url.toString(), {
      headers: { authorization: `Bearer ${Redacted.value(requestToken)}` },
    })).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "GitHub token request failed" })))
    if (response.status !== 200) return yield* new LabApiError({ status: response.status, message: "GitHub token request rejected" })
    const body = yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(TokenResponse)),
      Effect.mapError(() => new LabApiError({ status: 0, message: "Invalid GitHub token response" })))
    // Expiry only controls local caching. The coordinator independently verifies the signature and identity.
    const claims = yield* Effect.try({ try: () => decodeJwt(body.value), catch: () => new LabApiError({ status: 0, message: "Malformed GitHub token" }) })
    const lifetime = yield* Schema.decodeUnknown(Lifetime)(claims).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "Missing GitHub token expiry" })))
    if (!Number.isFinite(lifetime.exp) || lifetime.exp * 1000 < (yield* Clock.currentTimeMillis) + 90_000) return yield* new LabApiError({ status: 0, message: "GitHub token expires too soon" })
    return Redacted.make(body.value)
  }).pipe(Effect.timeoutFail({ duration: "15 seconds", onTimeout: () => new LabApiError({ status: 0, message: "GitHub token request timed out" }) }))
  return yield* Effect.cachedWithTTL(acquire, "60 seconds")
})
export const configuredClientToken = Effect.gen(function* () {
  const mode = yield* Config.literal("bearer", "github", "entra")("LAB_AUTH").pipe(Config.withDefault("bearer"))
  if (mode === "bearer") return yield* Config.redacted("LAB_TOKEN").pipe(Effect.map(token => Effect.succeed(token)))
  const staticToken = yield* Config.option(Config.redacted("LAB_TOKEN"))
  if (Option.isSome(staticToken)) return yield* new LabApiError({ status: 0, message: "Remove LAB_TOKEN when using identity authentication" })
  if (mode === "entra") return yield* entraClientToken(
    yield* Config.string("LAB_ENTRA_TENANT").pipe(Effect.flatMap(Schema.decodeUnknown(EntraTenantId))),
    yield* Config.string("LAB_ENTRA_APPLICATION").pipe(Effect.flatMap(Schema.decodeUnknown(EntraApplicationId))),
  )
  return yield* githubClientToken(yield* Config.string("ACTIONS_ID_TOKEN_REQUEST_URL"),
    yield* Config.redacted("ACTIONS_ID_TOKEN_REQUEST_TOKEN"), yield* Config.string("LAB_OIDC_AUDIENCE"))
})
