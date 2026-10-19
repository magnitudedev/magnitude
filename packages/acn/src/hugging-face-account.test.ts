import { FetchHttpClient } from "@effect/platform"
import { RpcTest } from "@effect/rpc"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Clock, Duration, Effect, Encoding, Layer, Option, Redacted, Schedule, Stream } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtemp, rm, stat } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { HuggingFace, rpcGroup, type HuggingFaceAccount as HuggingFaceAccountState, type HuggingFaceSignInAttemptId } from "@magnitudedev/acn-protocol"
import {
  GlobalStorage,
  MagnitudeStorage,
  makeAuthStorage,
  makeGlobalStoragePaths,
  type HuggingFaceCredential,
  type MagnitudeStorageShape,
} from "@magnitudedev/storage"
import { AcnChanges } from "./changes"
import {
  HuggingFaceAccount,
  HuggingFaceAccountLive,
  HuggingFaceOAuthConfig,
  makePkce,
} from "./hugging-face-account"

type TokenReply = { readonly status: number; readonly body: Record<string, unknown> }

/** A stub Hugging Face authorization server that records what Magnitude sends. */
const startStubHub = () => {
  const tokenRequests: URLSearchParams[] = []
  const deviceRequests: URLSearchParams[] = []
  const userinfoAuthorizations: (string | null)[] = []
  let tokenReplies: TokenReply[] = []
  let fallbackReply: TokenReply = { status: 400, body: { error: "authorization_pending" } }
  let deviceReply: TokenReply = {
    status: 200,
    body: { device_code: "device-1", user_code: "ABCD-EFGH", verification_uri: "https://huggingface.co/oauth/device", expires_in: 600, interval: 1 },
  }
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      const url = new URL(request.url)
      if (url.pathname === "/oauth/device") {
        deviceRequests.push(new URLSearchParams(await request.text()))
        return Response.json(deviceReply.body, { status: deviceReply.status })
      }
      if (url.pathname === "/oauth/token") {
        tokenRequests.push(new URLSearchParams(await request.text()))
        const reply = tokenReplies.shift() ?? fallbackReply
        return Response.json(reply.body, { status: reply.status })
      }
      if (url.pathname === "/oauth/userinfo") {
        userinfoAuthorizations.push(request.headers.get("authorization"))
        return Response.json({ sub: "1", preferred_username: "ada", name: "Ada Lovelace", picture: "https://hf.co/avatars/ada.png" })
      }
      return new Response("not found", { status: 404 })
    },
  })
  return {
    url: `http://127.0.0.1:${server.port}`,
    tokenRequests,
    deviceRequests,
    userinfoAuthorizations,
    reply: (...replies: TokenReply[]) => { tokenReplies = [...tokenReplies, ...replies] },
    otherwise: (reply: TokenReply) => { fallbackReply = reply },
    device: (reply: TokenReply) => { deviceReply = reply },
    stop: () => server.stop(true),
  }
}

const granted = (accessToken: string, extra: Record<string, unknown> = {}): TokenReply => ({
  status: 200,
  body: { access_token: accessToken, token_type: "bearer", scope: "openid profile read-repos", ...extra },
})
const oauthError = (error: string): TokenReply => ({ status: 400, body: { error } })

/** Server-given seconds pass a hundred times faster, so polling and renewal run in milliseconds. */
const base = Clock.make()
const fastClock: Clock.Clock = {
  [Clock.ClockTypeId]: Clock.ClockTypeId,
  unsafeCurrentTimeMillis: () => base.unsafeCurrentTimeMillis(),
  currentTimeMillis: base.currentTimeMillis,
  unsafeCurrentTimeNanos: () => base.unsafeCurrentTimeNanos(),
  currentTimeNanos: base.currentTimeNanos,
  sleep: (duration) => base.sleep(Duration.millis(Duration.toMillis(duration) / 100)),
}

const waitFor = (account: Effect.Effect<HuggingFaceAccountState>, predicate: (state: HuggingFaceAccountState) => boolean) =>
  account.pipe(
    Effect.filterOrFail(predicate, () => "not yet"),
    Effect.retry(Schedule.spaced("5 millis")),
    Effect.timeout("5 seconds"),
    Effect.orDie,
    Effect.withClock(base),
  )

describe("Hugging Face sign-in", () => {
  let root: string
  let hub: ReturnType<typeof startStubHub>
  let published: string[]

  const run = <A, E>(program: Effect.Effect<A, E, HuggingFaceAccount | GlobalStorage>) => {
    const global = Layer.succeed(GlobalStorage, GlobalStorage.of({ root, paths: makeGlobalStoragePaths(root) }))
    const storage = Layer.effect(MagnitudeStorage, makeAuthStorage().pipe(
      Effect.map((auth) => ({ auth }) as unknown as MagnitudeStorageShape),
    )).pipe(Layer.provide(Layer.mergeAll(BunFileSystem.layer, BunPath.layer, global)))
    const changes = Layer.succeed(AcnChanges, {
      publish: (change) => Effect.sync(() => { published.push(change.operation) }),
      stream: Stream.empty,
    })
    const config = Layer.succeed(HuggingFaceOAuthConfig, {
      hubUrl: hub.url,
      clientId: "test-client",
      scopes: "openid profile read-repos",
      browserTimeout: Duration.seconds(30),
      refreshMargin: Duration.seconds(300),
    })
    const layer = HuggingFaceAccountLive.pipe(Layer.provide(Layer.mergeAll(storage, changes, config, FetchHttpClient.layer)))
    return Effect.runPromise(Effect.scoped(program.pipe(
      Effect.provide(Layer.merge(layer, global)),
      Effect.withClock(fastClock),
    )) as Effect.Effect<A, E, never>)
  }

  const storedCredential = Effect.gen(function* () {
    const auth = yield* makeAuthStorage()
    return yield* auth.getHuggingFace
  }).pipe(Effect.provide(Layer.merge(BunFileSystem.layer, BunPath.layer)))

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-hf-"))
    hub = startStubHub()
    published = []
  })

  afterEach(async () => {
    hub.stop()
    await rm(root, { recursive: true, force: true })
  })

  it("makes an S256 PKCE challenge from a high-entropy verifier", async () => {
    const pkce = await Effect.runPromise(makePkce)
    expect(pkce.verifier).toMatch(/^[A-Za-z0-9_-]{43}$/)
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(pkce.verifier)))
    expect(pkce.challenge).toBe(Encoding.encodeBase64Url(digest))
    expect((await Effect.runPromise(makePkce)).verifier).not.toBe(pkce.verifier)
  })

  it("completes a browser sign-in on a loopback redirect, checking state and sending the PKCE verifier", async () => {
    hub.reply(granted("hf_oauth_browser", { refresh_token: "hf_oauth_refresh", expires_in: 28_800 }))
    const result = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      const { authorizationUrl } = yield* account.startBrowserSignIn
      const authorize = new URL(authorizationUrl)
      const redirect = new URL(authorize.searchParams.get("redirect_uri")!)
      const forged = yield* Effect.promise(() => fetch(`${redirect.href}?code=forged&state=wrong`))
      const stillSigningIn = yield* account.account
      const page = yield* Effect.promise(() =>
        fetch(`${redirect.href}?code=the-code&state=${authorize.searchParams.get("state")}`).then(async (response) => ({ status: response.status, text: await response.text() })))
      return { authorize, redirect, forgedStatus: forged.status, stillSigningIn, page, state: yield* account.account, token: yield* account.token }
    }))
    expect(result.authorize.origin + result.authorize.pathname).toBe(`${hub.url}/oauth/authorize`)
    expect(Object.fromEntries(result.authorize.searchParams)).toMatchObject({
      client_id: "test-client",
      response_type: "code",
      scope: "openid profile read-repos",
      code_challenge_method: "S256",
    })
    expect(result.redirect.hostname).toBe("127.0.0.1")
    expect(result.redirect.pathname).toBe("/callback")
    expect(result.forgedStatus).toBe(400)
    expect(result.stillSigningIn).toEqual({ _tag: "SigningIn", method: "Browser" })
    expect(result.page).toMatchObject({ status: 200 })
    expect(result.page.text).toContain("Signed in to Magnitude")
    expect(hub.tokenRequests).toHaveLength(1)
    const exchange = hub.tokenRequests[0]!
    expect(exchange.get("grant_type")).toBe("authorization_code")
    expect(exchange.get("code")).toBe("the-code")
    expect(exchange.get("redirect_uri")).toBe(result.redirect.href)
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(exchange.get("code_verifier")!)))
    expect(Encoding.encodeBase64Url(digest)).toBe(result.authorize.searchParams.get("code_challenge"))
    expect(hub.userinfoAuthorizations).toEqual(["Bearer hf_oauth_browser"])
    expect(result.state).toEqual({ _tag: "SignedIn", username: "ada", avatarUrl: Option.some("https://hf.co/avatars/ada.png") })
    expect(JSON.stringify(result.state)).not.toContain("hf_oauth")
    expect(Option.map(result.token, Redacted.value)).toEqual(Option.some("hf_oauth_browser"))
    expect(published).toContain("GetHuggingFaceAccount")
    const saved = await Effect.runPromise(storedCredential.pipe(Effect.provideService(GlobalStorage, { root, paths: makeGlobalStoragePaths(root) })))
    expect(Option.map(saved, (credential: HuggingFaceCredential) => Redacted.value(credential.accessToken))).toEqual(Option.some("hf_oauth_browser"))
    expect((await stat(makeGlobalStoragePaths(root).authFile)).mode & 0o777).toBe(0o600)
  })

  it("signs out with Denied when the person declines in the browser", async () => {
    const state = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      const authorize = new URL((yield* account.startBrowserSignIn).authorizationUrl)
      const redirect = authorize.searchParams.get("redirect_uri")!
      const page = yield* Effect.promise(() => fetch(`${redirect}?error=access_denied&state=${authorize.searchParams.get("state")}`))
      expect(page.status).toBe(400)
      return yield* waitFor(account.account, (current) => current._tag === "SignedOut")
    }))
    expect(state).toEqual({ _tag: "SignedOut", reason: Option.some("Denied") })
    expect(hub.tokenRequests).toHaveLength(0)
  })

  it("polls a device sign-in through authorization_pending and slow_down until it is granted", async () => {
    hub.reply(oauthError("authorization_pending"), oauthError("slow_down"), oauthError("authorization_pending"), granted("hf_oauth_device"))
    const result = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      const device = yield* account.startDeviceSignIn
      const signingIn = yield* account.account
      const state = yield* waitFor(account.account, (current) => current._tag === "SignedIn")
      return { device, signingIn, state }
    }))
    expect(result.device).toMatchObject({ verificationUri: "https://huggingface.co/oauth/device", userCode: "ABCD-EFGH", expiresIn: 600, interval: 1 })
    expect(result.device.verificationUriComplete).toEqual(Option.none())
    expect(result.signingIn).toEqual({ _tag: "SigningIn", method: "Device" })
    expect(result.state).toMatchObject({ _tag: "SignedIn", username: "ada" })
    expect(hub.deviceRequests[0]?.get("client_id")).toBe("test-client")
    expect(hub.deviceRequests[0]?.get("scope")).toBe("openid profile read-repos")
    expect(hub.tokenRequests).toHaveLength(4)
    for (const request of hub.tokenRequests) {
      expect(request.get("grant_type")).toBe("urn:ietf:params:oauth:grant-type:device_code")
      expect(request.get("device_code")).toBe("device-1")
    }
  })

  it("passes on the verification link with the code filled in when Hugging Face returns one", async () => {
    hub.device({ status: 200, body: { device_code: "d", user_code: "U", verification_uri: "https://huggingface.co/oauth/device", verification_uri_complete: "https://huggingface.co/oauth/device?user_code=U", expires_in: 600, interval: 1 } })
    const device = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      const started = yield* account.startDeviceSignIn
      yield* account.cancelSignIn(Option.none())
      return started
    }))
    expect(device.verificationUriComplete).toEqual(Option.some("https://huggingface.co/oauth/device?user_code=U"))
  })

  it("signs out with Expired when the device code expires or Hugging Face says so", async () => {
    hub.device({ status: 200, body: { device_code: "d", user_code: "U", verification_uri: "https://huggingface.co/oauth/device", expires_in: 3, interval: 1 } })
    const timedOut = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      yield* account.startDeviceSignIn
      return yield* waitFor(account.account, (current) => current._tag === "SignedOut")
    }))
    expect(timedOut).toEqual({ _tag: "SignedOut", reason: Option.some("Expired") })
    hub.reply(oauthError("expired_token"))
    const refused = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      yield* account.startDeviceSignIn
      return yield* waitFor(account.account, (current) => current._tag === "SignedOut")
    }))
    expect(refused).toEqual({ _tag: "SignedOut", reason: Option.some("Expired") })
  })

  it("stops polling when the sign-in is cancelled", async () => {
    const state = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      yield* account.startDeviceSignIn
      yield* account.cancelSignIn(Option.none())
      const polled = hub.tokenRequests.length
      yield* Effect.sleep("100 millis").pipe(Effect.withClock(base))
      expect(hub.tokenRequests.length).toBe(polled)
      return yield* account.account
    }))
    expect(state).toEqual({ _tag: "SignedOut", reason: Option.none() })
  })

  it("renews the token before it expires and hands the new one on", async () => {
    hub.reply(granted("hf_oauth_first", { refresh_token: "hf_oauth_r1", expires_in: 301 }), granted("hf_oauth_second", { expires_in: 28_800 }))
    const tokens = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      yield* account.startDeviceSignIn
      return yield* account.tokenChanges.pipe(
        Stream.filterMap((token) => Option.map(token, Redacted.value)),
        Stream.take(2),
        Stream.runCollect,
        Effect.timeout("500 seconds"),
      )
    }))
    expect([...tokens]).toEqual(["hf_oauth_first", "hf_oauth_second"])
    const refresh = hub.tokenRequests.at(-1)!
    expect(refresh.get("grant_type")).toBe("refresh_token")
    expect(refresh.get("refresh_token")).toBe("hf_oauth_r1")
    expect(refresh.get("client_id")).toBe("test-client")
  })

  it("signs out with SessionExpired when renewal is refused, and forgets the credential", async () => {
    hub.reply(granted("hf_oauth_first", { refresh_token: "hf_oauth_r1", expires_in: 301 }), oauthError("invalid_grant"))
    const state = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      yield* account.startDeviceSignIn
      yield* waitFor(account.account, (current) => current._tag === "SignedIn")
      const signedOut = yield* waitFor(account.account, (current) => current._tag === "SignedOut")
      expect(Option.isNone(yield* account.token)).toBe(true)
      return signedOut
    }))
    expect(state).toEqual({ _tag: "SignedOut", reason: Option.some("SessionExpired") })
    const saved = await Effect.runPromise(storedCredential.pipe(Effect.provideService(GlobalStorage, { root, paths: makeGlobalStoragePaths(root) })))
    expect(Option.isNone(saved)).toBe(true)
  })

  it("restores a saved account and signs out on request", async () => {
    hub.reply(granted("hf_oauth_saved"))
    await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      yield* account.startDeviceSignIn
      yield* waitFor(account.account, (current) => current._tag === "SignedIn")
    }))
    const result = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      const restored = yield* account.account
      const token = yield* account.token
      yield* account.signOut
      return { restored, token, after: yield* account.account, tokenAfter: yield* account.token }
    }))
    expect(result.restored).toMatchObject({ _tag: "SignedIn", username: "ada" })
    expect(Option.map(result.token, Redacted.value)).toEqual(Option.some("hf_oauth_saved"))
    expect(result.after).toEqual({ _tag: "SignedOut", reason: Option.none() })
    expect(Option.isNone(result.tokenAfter)).toBe(true)
  })

  it("starts each attempt without the reason an earlier one left, and follows its own outcome", async () => {
    hub.reply(oauthError("expired_token"))
    const result = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      const first = yield* account.startDeviceSignIn
      const expired = yield* waitFor(account.account, (current) => current._tag === "SignedOut")
      const firstOutcome = yield* account.attemptOutcome(first.attemptId)
      hub.otherwise(oauthError("authorization_pending"))
      const second = yield* account.startDeviceSignIn
      yield* Effect.sleep("100 millis").pipe(Effect.withClock(base))
      const pending = yield* account.account
      const secondOutcome = yield* account.attemptOutcome(second.attemptId)
      yield* account.cancelSignIn(Option.some(first.attemptId))
      const stillPending = yield* account.account
      yield* account.cancelSignIn(Option.some(second.attemptId))
      return { expired, firstOutcome, pending, secondOutcome, stillPending, cancelled: yield* account.attemptOutcome(second.attemptId), after: yield* account.account }
    }))
    expect(result.expired).toEqual({ _tag: "SignedOut", reason: Option.some("Expired") })
    expect(result.firstOutcome).toEqual({ _tag: "Failed", reason: "Expired" })
    expect(result.pending).toEqual({ _tag: "SigningIn", method: "Device" })
    expect(result.secondOutcome).toEqual({ _tag: "Pending" })
    expect(result.stillPending).toEqual({ _tag: "SigningIn", method: "Device" })
    expect(result.cancelled).toEqual({ _tag: "Cancelled" })
    expect(result.after).toEqual({ _tag: "SignedOut", reason: Option.none() })
  })

  it("reports a failed sign-in as failed even while an earlier account stays signed in", async () => {
    hub.reply(granted("hf_oauth_first"))
    const result = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      const first = yield* account.startDeviceSignIn
      yield* waitFor(account.account, (current) => current._tag === "SignedIn")
      hub.reply(oauthError("access_denied"))
      const second = yield* account.startDeviceSignIn
      const outcome = yield* account.attemptOutcome(second.attemptId).pipe(
        Effect.filterOrFail((current) => current._tag !== "Pending", () => "pending"),
        Effect.retry(Schedule.spaced("5 millis")),
        Effect.timeout("5 seconds"),
        Effect.withClock(base),
      )
      return { outcome, firstOutcome: yield* account.attemptOutcome(first.attemptId), state: yield* account.account, unknown: yield* account.attemptOutcome("nope" as never) }
    }))
    expect(result.firstOutcome).toEqual({ _tag: "SignedIn", username: "ada" })
    expect(result.outcome).toEqual({ _tag: "Failed", reason: "Denied" })
    expect(result.state).toMatchObject({ _tag: "SignedIn", username: "ada" })
    expect(result.unknown).toEqual({ _tag: "Unknown" })
  })

  it("marks an attempt superseded when a newer one starts", async () => {
    const outcome = await run(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      const first = yield* account.startDeviceSignIn
      yield* account.startDeviceSignIn
      return yield* account.attemptOutcome(first.attemptId)
    }))
    expect(outcome).toEqual({ _tag: "Superseded" })
  })

  it("keeps polling after the RPC that started it has returned", async () => {
    const group = rpcGroup({ huggingFace: HuggingFace })
    const handlers = group.toLayer(Effect.gen(function* () {
      const account = yield* HuggingFaceAccount
      return {
        GetHuggingFaceAccount: () => account.account,
        StartHuggingFaceSignIn: () => account.startBrowserSignIn,
        StartHuggingFaceDeviceSignIn: () => account.startDeviceSignIn,
        GetHuggingFaceSignInAttempt: ({ attemptId }: { attemptId: HuggingFaceSignInAttemptId }) => account.attemptOutcome(attemptId),
        CancelHuggingFaceSignIn: ({ attemptId }: { attemptId: Option.Option<HuggingFaceSignInAttemptId> }) => account.cancelSignIn(attemptId).pipe(Effect.as({})),
        SignOutOfHuggingFace: () => account.signOut.pipe(Effect.as({})),
      } as never
    }))
    const result = await run(Effect.gen(function* () {
      const context = yield* Layer.build(handlers)
      const client = yield* RpcTest.makeClient(group).pipe(Effect.provide(context))
      const device = yield* client.StartHuggingFaceDeviceSignIn({})
      const polledAtStart = hub.tokenRequests.length
      yield* Effect.sleep("300 millis").pipe(Effect.withClock(base))
      return { polled: hub.tokenRequests.length - polledAtStart, state: yield* client.GetHuggingFaceAccount({}), outcome: yield* client.GetHuggingFaceSignInAttempt({ attemptId: device.attemptId }) }
    }) as Effect.Effect<{ polled: number; state: unknown; outcome: unknown }, unknown, HuggingFaceAccount>)
    expect(result.polled).toBeGreaterThan(3)
    expect(result.state).toEqual({ _tag: "SigningIn", method: "Device" })
    expect(result.outcome).toEqual({ _tag: "Pending" })
  })
})
