import { HttpClient, HttpClientRequest, HttpServer, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import { BunHttpServer } from "@effect/platform-bun"
import {
  Clock,
  Context,
  Data,
  Deferred,
  Duration,
  Effect,
  Encoding,
  FiberHandle,
  Layer,
  Option,
  Redacted,
  Ref,
  Schedule,
  Schema,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  HuggingFace,
  HuggingFaceSignInFailed,
  type HuggingFaceAccount as HuggingFaceAccountState,
  type HuggingFaceBrowserSignIn,
  type HuggingFaceDeviceSignIn,
  type HuggingFaceSignedOutReason,
  type HuggingFaceSignInAttemptId,
  type HuggingFaceSignInMethod,
  type HuggingFaceSignInOutcome,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage, type HuggingFaceCredential } from "@magnitudedev/storage"
import { AcnChanges } from "./changes"

/** Magnitude's public Hugging Face OAuth app: no secret, loopback redirects on any port. */
export interface HuggingFaceOAuthSettings {
  readonly hubUrl: string
  readonly clientId: string
  readonly scopes: string
  readonly browserTimeout: Duration.Duration
  readonly refreshMargin: Duration.Duration
}

export class HuggingFaceOAuthConfig extends Context.Tag("HuggingFaceOAuthConfig")<
  HuggingFaceOAuthConfig,
  HuggingFaceOAuthSettings
>() {}

export const HUGGING_FACE_OAUTH: HuggingFaceOAuthSettings = {
  hubUrl: "https://huggingface.co",
  clientId: "de4f3e7d-8d7a-4028-9b23-259773b49623",
  scopes: "openid profile read-repos",
  browserTimeout: Duration.minutes(10),
  refreshMargin: Duration.minutes(5),
}

export const HuggingFaceOAuthConfigLive = Layer.succeed(HuggingFaceOAuthConfig, HUGGING_FACE_OAUTH)

/**
 * The one Hugging Face sign-in shared by every client. ACN completes a browser (authorization code
 * with PKCE, redirected to a loopback port) or device-code sign-in, keeps the credential in
 * `auth.json`, renews it before it expires, and exposes only the account, never the token, to
 * clients. `token` and `tokenChanges` are for delivering the credential to the inference service.
 */
export interface HuggingFaceAccountApi {
  readonly account: Effect.Effect<HuggingFaceAccountState>
  readonly token: Effect.Effect<Option.Option<Redacted.Redacted<string>>>
  readonly tokenChanges: Stream.Stream<Option.Option<Redacted.Redacted<string>>>
  readonly startBrowserSignIn: Effect.Effect<HuggingFaceBrowserSignIn, HuggingFaceSignInFailed>
  readonly startDeviceSignIn: Effect.Effect<HuggingFaceDeviceSignIn, HuggingFaceSignInFailed>
  readonly attemptOutcome: (attemptId: HuggingFaceSignInAttemptId) => Effect.Effect<HuggingFaceSignInOutcome>
  readonly cancelSignIn: (attemptId: Option.Option<HuggingFaceSignInAttemptId>) => Effect.Effect<void>
  readonly signOut: Effect.Effect<void, HuggingFaceSignInFailed>
}

export class HuggingFaceAccount extends Context.Tag("HuggingFaceAccount")<
  HuggingFaceAccount,
  HuggingFaceAccountApi
>() {}

interface SessionState {
  readonly credential: Option.Option<HuggingFaceCredential>
  readonly attempt: Option.Option<{ readonly id: HuggingFaceSignInAttemptId; readonly method: HuggingFaceSignInMethod }>
  readonly signedOutReason: Option.Option<HuggingFaceSignedOutReason>
}

export const accountOf = (state: SessionState): HuggingFaceAccountState => {
  if (Option.isSome(state.attempt)) return { _tag: "SigningIn", method: state.attempt.value.method }
  return Option.match(state.credential, {
    onNone: () => ({ _tag: "SignedOut", reason: state.signedOutReason }),
    onSome: (credential) => ({ _tag: "SignedIn", username: credential.username, avatarUrl: credential.avatarUrl }),
  })
}

/** The authorization server answered with an OAuth error code. */
class OAuthRejected extends Data.TaggedError("OAuthRejected")<{ readonly error: string }> {}
/** The authorization server couldn't be reached or answered something unexpected. */
class OAuthUnavailable extends Data.TaggedError("OAuthUnavailable")<{ readonly detail: string }> {}
type OAuthFailure = OAuthRejected | OAuthUnavailable

const TokenResponse = Schema.Struct({
  access_token: Schema.String,
  refresh_token: Schema.optional(Schema.String),
  expires_in: Schema.optional(Schema.Number),
})
type TokenResponse = typeof TokenResponse.Type

const OAuthErrorResponse = Schema.Struct({ error: Schema.String })

const DeviceAuthorization = Schema.Struct({
  device_code: Schema.String,
  user_code: Schema.String,
  verification_uri: Schema.String,
  verification_uri_complete: Schema.optional(Schema.String),
  expires_in: Schema.Number,
  interval: Schema.optional(Schema.Number),
})

const UserInfo = Schema.Struct({
  preferred_username: Schema.optional(Schema.String),
  name: Schema.optional(Schema.String),
  picture: Schema.optional(Schema.String),
})

const DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

const randomToken = (bytes: number): string =>
  Encoding.encodeBase64Url(crypto.getRandomValues(new Uint8Array(bytes)))

/** A PKCE pair (RFC 7636): a high-entropy verifier and its S256 challenge. */
export const makePkce = Effect.promise(async () => {
  const verifier = randomToken(32)
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier))
  return { verifier, challenge: Encoding.encodeBase64Url(new Uint8Array(digest)) }
})

const callbackPage = (title: string, detail: string, status: number) => HttpServerResponse.text(
  `<!doctype html><html><head><meta charset="utf-8"><title>${title}</title></head>`
    + `<body style="font-family: system-ui, sans-serif; display: grid; place-items: center; min-height: 90vh; margin: 0">`
    + `<main style="text-align: center"><h1 style="font-size: 20px; font-weight: 600">${title}</h1>`
    + `<p style="color: #666">${detail}</p></main></body></html>`,
  { status, contentType: "text/html; charset=utf-8" },
)

/** How many finished attempts ACN remembers for clients following their own attempt. */
const REMEMBERED_ATTEMPTS = 16

const reasonOf = (failure: OAuthFailure): "Denied" | "Expired" | "Failed" => {
  if (failure._tag === "OAuthUnavailable") return "Failed"
  switch (failure.error) {
    case "access_denied": return "Denied"
    case "expired_token": return "Expired"
    default: return "Failed"
  }
}

const signInFailed = (message: string) => new HuggingFaceSignInFailed({ message })

export const HuggingFaceAccountLive: Layer.Layer<
  HuggingFaceAccount,
  never,
  MagnitudeStorage | AcnChanges | HttpClient.HttpClient | HuggingFaceOAuthConfig
> = Layer.scoped(HuggingFaceAccount, Effect.gen(function* () {
  const storage = yield* MagnitudeStorage
  const changes = yield* AcnChanges
  const http = yield* HttpClient.HttpClient
  const settings = yield* HuggingFaceOAuthConfig
  const stored = yield* storage.auth.getHuggingFace.pipe(
    Effect.tapError((error) => Effect.logWarning("Could not read the Hugging Face account from auth.json; signed out").pipe(
      Effect.annotateLogs({ cause: error.message }),
    )),
    Effect.orElseSucceed(() => Option.none<HuggingFaceCredential>()),
  )
  const state = yield* SubscriptionRef.make<SessionState>({
    credential: stored,
    attempt: Option.none(),
    signedOutReason: Option.none(),
  })
  const flow = yield* FiberHandle.make<void, never>()
  const outcomes = yield* Ref.make<ReadonlyMap<HuggingFaceSignInAttemptId, HuggingFaceSignInOutcome>>(new Map())
  const renewal = yield* FiberHandle.make<void, never>()

  const update = (f: (current: SessionState) => SessionState) =>
    SubscriptionRef.update(state, f).pipe(
      Effect.zipRight(changes.publish({ operation: HuggingFace.getHuggingFaceAccount._tag })),
    )

  const endpoint = (path: string) => `${settings.hubUrl}${path}`

  const postForm = <A, I>(path: string, form: Record<string, string>, schema: Schema.Schema<A, I>) =>
    Effect.gen(function* () {
      const response = yield* http.execute(HttpClientRequest.post(endpoint(path)).pipe(
        HttpClientRequest.acceptJson,
        HttpClientRequest.bodyUrlParams(form),
      )).pipe(Effect.mapError((error) => new OAuthUnavailable({ detail: error.message })))
      const body = yield* response.json.pipe(Effect.orElseSucceed(() => undefined as unknown))
      if (response.status >= 200 && response.status < 300) {
        return yield* Schema.decodeUnknown(schema)(body).pipe(
          Effect.mapError(() => new OAuthUnavailable({ detail: `unexpected response from ${path}` })),
        )
      }
      const rejected = Schema.decodeUnknownOption(OAuthErrorResponse)(body)
      return yield* Option.match(rejected, {
        onNone: () => new OAuthUnavailable({ detail: `${path} returned HTTP ${response.status}` }),
        onSome: ({ error }) => new OAuthRejected({ error }),
      })
    }).pipe(Effect.scoped)

  const userInfo = (accessToken: Redacted.Redacted<string>) => Effect.gen(function* () {
    const response = yield* http.execute(HttpClientRequest.get(endpoint("/oauth/userinfo")).pipe(
      HttpClientRequest.acceptJson,
      HttpClientRequest.bearerToken(Redacted.value(accessToken)),
    ))
    if (response.status !== 200) return yield* new OAuthUnavailable({ detail: `userinfo returned HTTP ${response.status}` })
    return yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(UserInfo)))
  }).pipe(
    Effect.scoped,
    Effect.catchAll((error) => error instanceof OAuthUnavailable
      ? Effect.fail(error)
      : Effect.fail(new OAuthUnavailable({ detail: "userinfo could not be read" }))),
  )

  const credentialFrom = (tokens: TokenResponse, previous: Option.Option<HuggingFaceCredential>) =>
    Effect.gen(function* () {
      const now = yield* Clock.currentTimeMillis
      const accessToken = Redacted.make(tokens.access_token)
      const identity = yield* Option.match(previous, {
        onNone: () => userInfo(accessToken).pipe(Effect.map((info) => ({
          username: info.preferred_username ?? info.name ?? "Hugging Face account",
          avatarUrl: Option.fromNullable(info.picture),
        }))),
        onSome: (credential) => Effect.succeed({ username: credential.username, avatarUrl: credential.avatarUrl }),
      })
      return {
        type: "huggingface-oauth",
        accessToken,
        refreshToken: Option.orElse(
          Option.map(Option.fromNullable(tokens.refresh_token), Redacted.make),
          () => Option.flatMap(previous, (credential) => credential.refreshToken),
        ),
        expiresAt: Option.map(Option.fromNullable(tokens.expires_in), (seconds) => now + seconds * 1000),
        ...identity,
      } satisfies HuggingFaceCredential
    })

  const signedOut = (reason: Option.Option<HuggingFaceSignedOutReason>) =>
    storage.auth.removeHuggingFace.pipe(
      Effect.tapError((error) => Effect.logWarning("Could not remove the Hugging Face account from auth.json").pipe(
        Effect.annotateLogs({ cause: error.message }),
      )),
      Effect.ignore,
      Effect.zipRight(update((current) => ({ ...current, credential: Option.none(), signedOutReason: reason }))),
    )

  // Renews the access token `refreshMargin` before it expires. A refresh Hugging Face refuses signs
  // out with `SessionExpired`; while it can't be reached the refresh is retried, backing off to every
  // five minutes. A credential without a refresh token signs out when it expires.
  const renew = (credential: HuggingFaceCredential): Effect.Effect<void> => Option.match(credential.expiresAt, {
    onNone: () => Effect.void,
    onSome: (expiresAt) => Effect.gen(function* () {
      const now = yield* Clock.currentTimeMillis
      const margin = Option.isSome(credential.refreshToken) ? Duration.toMillis(settings.refreshMargin) : 0
      yield* Effect.sleep(Duration.millis(Math.max(0, expiresAt - margin - now)))
      if (Option.isNone(credential.refreshToken)) return yield* signedOut(Option.some("SessionExpired"))
      const refreshToken = credential.refreshToken.value
      const refreshed = yield* postForm("/oauth/token", {
        grant_type: "refresh_token",
        refresh_token: Redacted.value(refreshToken),
        client_id: settings.clientId,
      }, TokenResponse).pipe(
        Effect.retry({
          schedule: Schedule.exponential("5 seconds").pipe(Schedule.union(Schedule.spaced("5 minutes"))),
          while: (failure) => failure._tag === "OAuthUnavailable",
        }),
        Effect.flatMap((tokens) => credentialFrom(tokens, Option.some(credential))),
        Effect.option,
      )
      if (Option.isNone(refreshed)) {
        yield* Effect.logWarning("The Hugging Face session could not be renewed; signed out")
        return yield* signedOut(Option.some("SessionExpired"))
      }
      yield* storage.auth.setHuggingFace(refreshed.value).pipe(
        Effect.tapError((error) => Effect.logWarning("Could not save the renewed Hugging Face account").pipe(
          Effect.annotateLogs({ cause: error.message }),
        )),
        Effect.ignore,
      )
      yield* update((current) => ({ ...current, credential: refreshed }))
      yield* renew(refreshed.value)
    }),
  })

  const signedIn = (credential: HuggingFaceCredential) => storage.auth.setHuggingFace(credential).pipe(
    Effect.mapError((error) => new OAuthUnavailable({ detail: `auth.json could not be written: ${error.message}` })),
    Effect.zipRight(update((current) => ({
      ...current,
      credential: Option.some(credential),
      signedOutReason: Option.none(),
    }))),
    Effect.zipRight(FiberHandle.run(renewal, renew(credential))),
    Effect.asVoid,
  )

  const recordOutcome = (id: HuggingFaceSignInAttemptId, outcome: HuggingFaceSignInOutcome) =>
    Ref.update(outcomes, (current) => {
      const next = new Map(current)
      next.delete(id)
      next.set(id, outcome)
      while (next.size > REMEMBERED_ATTEMPTS) next.delete(next.keys().next().value!)
      return next
    })

  // Ends attempt `id` if it is still the one in progress: records how it ended, clears it from the
  // account and, for a failure, says why the account is signed out.
  const finish = (id: HuggingFaceSignInAttemptId, outcome: HuggingFaceSignInOutcome) =>
    Effect.gen(function* () {
      const current = yield* Ref.get(outcomes)
      if (current.get(id)?._tag !== "Pending") return
      yield* recordOutcome(id, outcome)
      yield* update((state) => Option.exists(state.attempt, (attempt) => attempt.id === id)
        ? {
            ...state,
            attempt: Option.none(),
            signedOutReason: outcome._tag === "Failed" ? Option.some(outcome.reason) : state.signedOutReason,
          }
        : state)
    })

  const attemptFailed = (id: HuggingFaceSignInAttemptId) => (failure: OAuthFailure) =>
    Effect.logWarning("Hugging Face sign-in did not complete").pipe(
      Effect.annotateLogs({ reason: failure._tag === "OAuthRejected" ? failure.error : failure.detail }),
      Effect.zipRight(finish(id, { _tag: "Failed", reason: reasonOf(failure) })),
    )

  // A new attempt supersedes the one in progress and clears the reason an earlier one left.
  const begin = (method: HuggingFaceSignInMethod) => Effect.gen(function* () {
    const previous = (yield* SubscriptionRef.get(state)).attempt
    if (Option.isSome(previous)) yield* finish(previous.value.id, { _tag: "Superseded" })
    yield* FiberHandle.clear(flow)
    const id = randomToken(12) as HuggingFaceSignInAttemptId
    yield* recordOutcome(id, { _tag: "Pending" })
    yield* update((current) => ({ ...current, attempt: Option.some({ id, method }), signedOutReason: Option.none() }))
    return id
  })

  // A flow runs detached in `flow`, owned by this service's scope rather than the request that
  // started it, until it signs in, fails, expires, or is cancelled or superseded.
  const runFlow = (id: HuggingFaceSignInAttemptId, body: Effect.Effect<HuggingFaceCredential, OAuthFailure>) =>
    FiberHandle.run(flow, body.pipe(
      Effect.matchEffect({
        onFailure: attemptFailed(id),
        onSuccess: (credential) => finish(id, { _tag: "SignedIn", username: credential.username }),
      }),
      Effect.onInterrupt(() => finish(id, { _tag: "Cancelled" })),
    ))

  const exchange = (form: Record<string, string>) => postForm("/oauth/token", {
    ...form,
    client_id: settings.clientId,
  }, TokenResponse).pipe(
    Effect.flatMap((tokens) => credentialFrom(tokens, Option.none())),
    Effect.tap(signedIn),
  )

  const startBrowserSignIn = Effect.gen(function* () {
    const attemptId = yield* begin("Browser")
    const pkce = yield* makePkce
    const expectedState = randomToken(16)
    const authorization = yield* Deferred.make<string, OAuthFailure>()
    const callback = yield* Deferred.make<{ readonly code: Option.Option<string>; readonly error: Option.Option<string> }>()
    const outcome = yield* Deferred.make<boolean>()
    const app = Effect.gen(function* () {
      const request = yield* HttpServerRequest.HttpServerRequest
      const url = new URL(request.url, "http://127.0.0.1")
      if (url.pathname !== "/callback") return HttpServerResponse.text("Not found", { status: 404 })
      if (url.searchParams.get("state") !== expectedState) {
        return callbackPage("This sign-in link is out of date", "Start signing in to Hugging Face again from Magnitude.", 400)
      }
      yield* Deferred.succeed(callback, {
        code: Option.fromNullable(url.searchParams.get("code")),
        error: Option.fromNullable(url.searchParams.get("error")),
      })
      const succeeded = yield* Deferred.await(outcome).pipe(
        Effect.timeout("30 seconds"),
        Effect.orElseSucceed(() => false),
      )
      return succeeded
        ? callbackPage("Signed in to Magnitude", "You can close this tab.", 200)
        : callbackPage("Magnitude couldn’t sign in to Hugging Face", "Close this tab and try again from Magnitude.", 400)
    })
    const body = Effect.gen(function* () {
      const server = yield* BunHttpServer.make({ hostname: "127.0.0.1", port: 0 })
      yield* server.serve(app)
      if (server.address._tag !== "TcpAddress") return yield* new OAuthUnavailable({ detail: "no loopback port" })
      const redirectUri = `http://127.0.0.1:${server.address.port}/callback`
      const query = new URLSearchParams({
        client_id: settings.clientId,
        redirect_uri: redirectUri,
        response_type: "code",
        scope: settings.scopes,
        state: expectedState,
        code_challenge: pkce.challenge,
        code_challenge_method: "S256",
      })
      yield* Deferred.succeed(authorization, `${endpoint("/oauth/authorize")}?${query.toString()}`)
      const received = yield* Deferred.await(callback).pipe(
        Effect.timeoutFail({ duration: settings.browserTimeout, onTimeout: () => new OAuthRejected({ error: "expired_token" }) }),
      )
      return yield* Effect.gen(function* () {
        const code = yield* Option.match(received.code, {
          onNone: () => Effect.fail(new OAuthRejected({ error: Option.getOrElse(received.error, () => "invalid_request") })),
          onSome: Effect.succeed,
        })
        return yield* exchange({
          grant_type: "authorization_code",
          code,
          redirect_uri: redirectUri,
          code_verifier: pkce.verifier,
        })
      }).pipe(Effect.onExit((exit) => Deferred.succeed(outcome, exit._tag === "Success")))
    }).pipe(
      Effect.scoped,
      Effect.tapError((failure) => Deferred.fail(authorization, failure)),
      Effect.ensuring(Deferred.fail(authorization, new OAuthUnavailable({ detail: "the sign-in ended" }))),
    )
    yield* runFlow(attemptId, body)
    const authorizationUrl = yield* Deferred.await(authorization).pipe(
      Effect.mapError(() => signInFailed("Magnitude couldn’t start signing in to Hugging Face. Try again.")),
    )
    return { attemptId, authorizationUrl }
  })

  const startDeviceSignIn = Effect.gen(function* () {
    const attemptId = yield* begin("Device")
    const device = yield* postForm("/oauth/device", { client_id: settings.clientId, scope: settings.scopes }, DeviceAuthorization).pipe(
      Effect.tapError(attemptFailed(attemptId)),
      Effect.mapError((failure) => signInFailed(failure._tag === "OAuthUnavailable"
        ? "Magnitude couldn’t reach Hugging Face. Check your internet connection, then try again."
        : "Hugging Face didn’t start the sign-in. Try again.")),
    )
    const interval = Math.max(1, Math.ceil(device.interval ?? 5))
    const expiresIn = Math.max(1, Math.ceil(device.expires_in))
    // RFC 8628 polling: wait `interval` between attempts, add 5 seconds on `slow_down`, and keep
    // waiting on `authorization_pending` or an unreachable server until the code expires.
    const poll = (wait: number): Effect.Effect<HuggingFaceCredential, OAuthFailure> => Effect.sleep(Duration.seconds(wait)).pipe(
      Effect.zipRight(exchange({ grant_type: DEVICE_CODE_GRANT, device_code: device.device_code })),
      Effect.catchIf(
        (failure) => failure._tag === "OAuthUnavailable"
          || failure.error === "authorization_pending" || failure.error === "slow_down",
        (failure) => poll(failure._tag === "OAuthRejected" && failure.error === "slow_down" ? wait + 5 : wait),
      ),
    )
    yield* runFlow(attemptId, poll(interval).pipe(
      Effect.timeoutFail({ duration: Duration.seconds(expiresIn), onTimeout: () => new OAuthRejected({ error: "expired_token" }) }),
    ))
    return {
      attemptId,
      verificationUri: device.verification_uri,
      verificationUriComplete: Option.fromNullable(device.verification_uri_complete),
      userCode: device.user_code,
      expiresIn,
      interval,
    }
  })

  yield* Option.match(stored, {
    onNone: () => Effect.void,
    onSome: (credential) => FiberHandle.run(renewal, renew(credential)).pipe(Effect.asVoid),
  })

  const tokenOf = (current: SessionState) => Option.map(current.credential, (credential) => credential.accessToken)
  const sameToken = Option.getEquivalence<Redacted.Redacted<string>>((a, b) => Redacted.value(a) === Redacted.value(b))

  return HuggingFaceAccount.of({
    account: SubscriptionRef.get(state).pipe(Effect.map(accountOf)),
    token: SubscriptionRef.get(state).pipe(Effect.map(tokenOf)),
    tokenChanges: state.changes.pipe(Stream.map(tokenOf), Stream.changesWith(sameToken)),
    startBrowserSignIn,
    startDeviceSignIn,
    attemptOutcome: (attemptId) => Ref.get(outcomes).pipe(
      Effect.map((current) => current.get(attemptId) ?? { _tag: "Unknown" as const }),
    ),
    cancelSignIn: (attemptId) => Effect.gen(function* () {
      const current = (yield* SubscriptionRef.get(state)).attempt
      if (Option.isNone(current) || Option.exists(attemptId, (id) => id !== current.value.id)) return
      yield* finish(current.value.id, { _tag: "Cancelled" })
      yield* FiberHandle.clear(flow)
    }),
    signOut: SubscriptionRef.get(state).pipe(
      Effect.flatMap(({ attempt }) => Option.match(attempt, {
        onNone: () => Effect.void,
        onSome: ({ id }) => finish(id, { _tag: "Cancelled" }),
      })),
      Effect.zipRight(FiberHandle.clear(flow)),
      Effect.zipRight(FiberHandle.clear(renewal)),
      Effect.zipRight(storage.auth.removeHuggingFace),
      Effect.mapError(() => signInFailed("Magnitude couldn’t remove the saved Hugging Face sign-in. Try again.")),
      Effect.zipRight(update(() => ({ credential: Option.none(), attempt: Option.none(), signedOutReason: Option.none() }))),
    ),
  })
}))
