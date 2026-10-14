import { FileSystem, HttpLayerRouter, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import { Clock, Duration, Effect, Option, Ref, Schema } from "effect"
import { createHash, randomBytes, timingSafeEqual } from "node:crypto"
import {
  REMOTE_ACCESS_SESSION_PATH,
  REMOTE_ACCESS_SIGN_OUT_PATH,
  RemoteAccessStatus,
  SignInFailure,
  SignInRefused,
  SignInRequest,
  SignInThrottled,
} from "@magnitudedev/acn-protocol"
import { isLoopbackAddress, readStructuredFile, writeStructuredFileAtomic, type NetworkAccess } from "@magnitudedev/storage"

export const SESSION_COOKIE = "magnitude_session"
export const SESSION_IDLE_LIMIT = Duration.days(30)
export const REMOTE_SESSIONS_FILE = "remote-sessions.json"
const FREE_FAILURES = 5
const FIRST_WAIT = Duration.seconds(1)
const LONGEST_WAIT = Duration.minutes(1)
const LAST_USE_SAVE_INTERVAL = Duration.hours(1)

/** Who is calling: this machine, a browser that signed in with the key, or anyone else. */
export type CallerAccess = "Local" | "SignedIn" | "Anonymous"

interface Throttle {
  readonly failures: number
  readonly wait: Duration.Duration
  readonly allowedAt: number
}

/**
 * Remote browser sessions, keyed by a digest of their token. They outlive a restart through the
 * session store, which drops them when the key changes, so regenerating the key still signs every
 * browser out once it applies.
 */
export interface RemoteAccessApi {
  readonly network: NetworkAccess
  readonly access: (request: HttpServerRequest.HttpServerRequest) => Effect.Effect<CallerAccess>
  readonly signIn: (address: string, key: string) => Effect.Effect<string, SignInFailure>
  readonly signOut: (token: Option.Option<string>) => Effect.Effect<void>
}

/** Whether the request comes from this machine. With loopback binding every caller is local. */
export const isLocalCaller = (request: HttpServerRequest.HttpServerRequest, network: NetworkAccess) => Option.match(request.remoteAddress, {
  onNone: () => !network.enabled,
  onSome: isLoopbackAddress,
})

/** Whether a browser request was made by a page served from this same address. */
export const isSameOrigin = (request: HttpServerRequest.HttpServerRequest) => {
  const origin = request.headers.origin
  const host = request.headers.host
  if (origin === undefined || host === undefined) return false
  try {
    return new URL(origin).host === host.toLowerCase()
  } catch {
    return false
  }
}

const keyMatches = (presented: string, expected: string) => {
  const left = Buffer.from(presented)
  const right = Buffer.from(expected)
  return left.length === right.length && timingSafeEqual(left, right)
}

const sessionToken = (request: HttpServerRequest.HttpServerRequest) => Option.fromNullable(request.cookies[SESSION_COOKIE])
const digest = (value: string) => createHash("sha256").update(value).digest("base64url")

/** Signed-in sessions kept across restarts: token digests and when each was last used. */
export interface RemoteSessionStore {
  readonly load: Effect.Effect<ReadonlyMap<string, number>>
  readonly save: (sessions: ReadonlyMap<string, number>) => Effect.Effect<void>
}

export const memorySessionStore: RemoteSessionStore = { load: Effect.succeed(new Map()), save: () => Effect.void }

const PersistedSessions = Schema.Struct({
  keyDigest: Schema.String,
  sessions: Schema.Array(Schema.Struct({ tokenDigest: Schema.String, lastUsed: Schema.Number })),
})

/** Sessions in a private file, valid only while the Network access key they signed in with is. */
export const fileSessionStore = (path: string, network: NetworkAccess) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (Option.isNone(network.apiKey)) return memorySessionStore
  const keyDigest = digest(network.apiKey.value)
  const writes = yield* Effect.makeSemaphore(1)
  return {
    load: readStructuredFile(path, PersistedSessions).pipe(
      Effect.map(result => result._tag === "Present" && result.value.keyDigest === keyDigest
        ? new Map(result.value.sessions.map(session => [session.tokenDigest, session.lastUsed] as const))
        : new Map<string, number>()),
      Effect.tapError(error => Effect.logWarning("Signed-in browser sessions could not be read").pipe(Effect.annotateLogs({ cause: error.message }))),
      Effect.orElseSucceed(() => new Map<string, number>()),
      Effect.provideService(FileSystem.FileSystem, fs),
    ),
    save: sessions => writes.withPermits(1)(writeStructuredFileAtomic(path, PersistedSessions, {
      keyDigest,
      sessions: [...sessions].map(([tokenDigest, lastUsed]) => ({ tokenDigest, lastUsed })),
    }, { mode: 0o600 })).pipe(
      Effect.catchAll(error => Effect.logWarning("Signed-in browser sessions could not be saved").pipe(Effect.annotateLogs({ cause: error.message }))),
      Effect.provideService(FileSystem.FileSystem, fs),
    ),
  } satisfies RemoteSessionStore
})

export const makeRemoteAccess = (network: NetworkAccess, store: RemoteSessionStore = memorySessionStore) => Effect.gen(function* () {
  const started = yield* Clock.currentTimeMillis
  const sessions = yield* Ref.make(new Map([...yield* store.load].filter(([, lastUsed]) => started - lastUsed <= Duration.toMillis(SESSION_IDLE_LIMIT))))
  const savedAt = yield* Ref.make(started)
  const throttles = yield* Ref.make(new Map<string, Throttle>())
  const save = Effect.gen(function* () {
    yield* Ref.set(savedAt, yield* Clock.currentTimeMillis)
    yield* store.save(yield* Ref.get(sessions))
  })

  const verify = (token: Option.Option<string>) => Effect.gen(function* () {
    if (Option.isNone(token)) return false
    const key = digest(token.value)
    const now = yield* Clock.currentTimeMillis
    const valid = yield* Ref.modify(sessions, current => {
      const lastUsed = current.get(key)
      if (lastUsed === undefined) return [false, current]
      const next = new Map(current)
      if (now - lastUsed > Duration.toMillis(SESSION_IDLE_LIMIT)) {
        next.delete(key)
        return [false, next]
      }
      next.set(key, now)
      return [true, next]
    })
    // Last use only needs to survive a restart roughly, so it is saved at most hourly.
    if (now - (yield* Ref.get(savedAt)) > Duration.toMillis(LAST_USE_SAVE_INTERVAL)) yield* save
    return valid
  })

  const recordFailure = (address: string, now: number) => Ref.update(throttles, current => {
    const previous = current.get(address)
    const failures = (previous?.failures ?? 0) + 1
    const wait = failures < FREE_FAILURES ? Duration.zero
      : previous === undefined || Duration.isZero(previous.wait) ? FIRST_WAIT
      : Duration.min(Duration.times(previous.wait, 2), LONGEST_WAIT)
    return new Map(current).set(address, { failures, wait, allowedAt: now + Duration.toMillis(wait) })
  })

  return {
    network,
    access: request => isLocalCaller(request, network) ? Effect.succeed("Local" as const)
      : verify(sessionToken(request)).pipe(Effect.map(valid => valid ? "SignedIn" as const : "Anonymous" as const)),
    signIn: (address, key) => Effect.gen(function* () {
      if (!network.enabled) return yield* new SignInRefused({ reason: "NetworkAccessOff" })
      if (Option.isNone(network.apiKey)) return yield* new SignInRefused({ reason: "NoKey" })
      const now = yield* Clock.currentTimeMillis
      const throttle = (yield* Ref.get(throttles)).get(address)
      if (throttle !== undefined && now < throttle.allowedAt) {
        return yield* new SignInThrottled({ retryAfterSeconds: Math.ceil((throttle.allowedAt - now) / 1000) })
      }
      if (!keyMatches(key, network.apiKey.value)) {
        yield* recordFailure(address, now)
        return yield* new SignInRefused({ reason: "WrongKey" })
      }
      yield* Ref.update(throttles, current => {
        const next = new Map(current)
        next.delete(address)
        return next
      })
      const token = randomBytes(32).toString("base64url")
      yield* Ref.update(sessions, current => new Map(current).set(digest(token), now))
      yield* save
      return token
    }),
    signOut: token => Option.match(token, {
      onNone: () => Effect.void,
      onSome: value => Ref.update(sessions, current => {
        const next = new Map(current)
        next.delete(digest(value))
        return next
      }).pipe(Effect.zipRight(save)),
    }),
  } satisfies RemoteAccessApi
})

const isHttps = (request: HttpServerRequest.HttpServerRequest) =>
  request.headers["x-forwarded-proto"]?.split(",")[0]?.trim().toLowerCase() === "https" || request.url.startsWith("https:")

const withSessionCookie = (response: HttpServerResponse.HttpServerResponse, request: HttpServerRequest.HttpServerRequest, token: string, maxAge: Duration.Duration) =>
  HttpServerResponse.unsafeSetCookie(response, SESSION_COOKIE, token, {
    path: "/", httpOnly: true, sameSite: "strict", secure: isHttps(request), maxAge,
  })

const failureStatus = (failure: SignInFailure) => failure._tag === "SignInThrottled" ? 429
  : failure.reason === "WrongKey" ? 401
  : failure.reason === "CrossOrigin" ? 403
  : 409

const encodeStatus = Schema.encodeSync(RemoteAccessStatus)
const encodeFailure = Schema.encodeSync(SignInFailure)
const decodeSignIn = Schema.decodeUnknown(SignInRequest)

/** `GET` reports how this caller is admitted, `POST` signs in with the key, and `/end` signs out. */
export const installRemoteAccessRoutes = (router: HttpLayerRouter.HttpRouter, remote: RemoteAccessApi) => Effect.gen(function* () {
  yield* router.add("GET", REMOTE_ACCESS_SESSION_PATH, Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest
    const access = yield* remote.access(request)
    const status: RemoteAccessStatus = access === "Local" ? { _tag: "Local" }
      : access === "SignedIn" ? { _tag: "SignedIn" }
      : { _tag: "SignInRequired", keyConfigured: remote.network.enabled && Option.isSome(remote.network.apiKey) }
    const response = HttpServerResponse.unsafeJson(encodeStatus(status), { headers: { "cache-control": "no-store" } })
    // Each visit renews the cookie, so a browser in regular use stays signed in.
    const token = sessionToken(request)
    return access === "SignedIn" && Option.isSome(token) ? withSessionCookie(response, request, token.value, SESSION_IDLE_LIMIT) : response
  }))
  yield* router.add("POST", REMOTE_ACCESS_SESSION_PATH, Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest
    if (isLocalCaller(request, remote.network)) return HttpServerResponse.empty({ status: 204 })
    const result = yield* Effect.gen(function* () {
      if (!isSameOrigin(request)) return yield* new SignInRefused({ reason: "CrossOrigin" })
      const body = yield* request.json.pipe(Effect.flatMap(decodeSignIn), Effect.orElseFail(() => new SignInRefused({ reason: "WrongKey" })))
      return yield* remote.signIn(Option.getOrElse(request.remoteAddress, () => "unknown"), body.key)
    }).pipe(Effect.either)
    if (result._tag === "Left") {
      const failure = result.left
      return HttpServerResponse.unsafeJson(encodeFailure(failure), {
        status: failureStatus(failure),
        headers: failure._tag === "SignInThrottled" ? { "retry-after": String(failure.retryAfterSeconds) } : {},
      })
    }
    return withSessionCookie(HttpServerResponse.empty({ status: 204 }), request, result.right, SESSION_IDLE_LIMIT)
  }))
  yield* router.add("POST", REMOTE_ACCESS_SIGN_OUT_PATH, Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest
    if (!isLocalCaller(request, remote.network) && !isSameOrigin(request)) return HttpServerResponse.empty({ status: 403 })
    yield* remote.signOut(sessionToken(request))
    return withSessionCookie(HttpServerResponse.empty({ status: 204 }), request, "", Duration.zero)
  }))
})
