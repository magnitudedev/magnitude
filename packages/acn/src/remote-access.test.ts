import { Duration, Effect, Either, Option, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { LOOPBACK_ONLY, type NetworkAccess } from "@magnitudedev/storage"
import { BunFileSystem } from "@effect/platform-bun"
import { mkdtempSync, readFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileSessionStore, makeRemoteAccess, memorySessionStore, SESSION_IDLE_LIMIT } from "./remote-access"

const network: NetworkAccess = {
  enabled: true, bind: "0.0.0.0", apiKey: Option.some("mag-key"), requireApiKey: true, allowedHosts: [], warning: Option.none(),
}
const run = <A, E>(effect: Effect.Effect<A, E>) => Effect.runPromise(effect.pipe(Effect.provide(TestContext.TestContext)))
const remoteRequest = (token?: string) => ({ remoteAddress: Option.some("192.168.1.20"), cookies: token === undefined ? {} : { magnitude_session: token } }) as never

describe("remote sign-in", () => {
  it("refuses remote sign-in and ignores persisted cookies when remote app access is disabled", async () => {
    const path = join(mkdtempSync(join(tmpdir(), "magnitude-remote-disabled-")), "remote-sessions.json")
    const result = await Effect.runPromise(Effect.gen(function* () {
      const enabled = yield* makeRemoteAccess(network, yield* fileSessionStore(path, network), true)
      const token = yield* enabled.signIn("10.0.0.2", "mag-key")
      const disabled = yield* makeRemoteAccess(network, yield* fileSessionStore(path, network), false)
      return {
        access: yield* disabled.access(remoteRequest(token)),
        signIn: yield* Effect.either(disabled.signIn("10.0.0.2", "mag-key")),
      }
    }).pipe(Effect.provide(BunFileSystem.layer)))
    expect(result.access).toBe("Anonymous")
    expect(Either.getLeft(result.signIn)).toEqual(Option.some(expect.objectContaining({ reason: "NetworkAccessOff" })))
  })

  it("refuses sign-in when Network access is off or no key is set", async () => {
    const [off, noKey] = await run(Effect.gen(function* () {
      const disabled = yield* makeRemoteAccess(LOOPBACK_ONLY, memorySessionStore, true)
      const keyless = yield* makeRemoteAccess({ ...network, apiKey: Option.none() }, memorySessionStore, true)
      return [yield* Effect.either(disabled.signIn("a", "mag-key")), yield* Effect.either(keyless.signIn("a", "mag-key"))] as const
    }))
    expect(Either.getLeft(off)).toEqual(Option.some(expect.objectContaining({ reason: "NetworkAccessOff" })))
    expect(Either.getLeft(noKey)).toEqual(Option.some(expect.objectContaining({ reason: "NoKey" })))
  })

  it("waits after five failures, doubling each time up to one minute, and resets on success", async () => {
    const timeline = await run(Effect.gen(function* () {
      const remote = yield* makeRemoteAccess(network, memorySessionStore, true)
      const attempt = (key: string) => remote.signIn("10.0.0.2", key).pipe(Effect.either, Effect.map(result =>
        Either.isRight(result) ? "ok" : result.left._tag === "SignInThrottled" ? `wait ${result.left.retryAfterSeconds}` : "wrong"))
      const steps: string[] = []
      for (let index = 0; index < 5; index++) steps.push(yield* attempt("bad"))
      steps.push(yield* attempt("bad"))
      steps.push(`other ${yield* remote.signIn("10.0.0.3", "bad").pipe(Effect.either, Effect.map(result => Either.isLeft(result) ? result.left._tag : "ok"))}`)
      for (const wait of [1, 2, 4, 8, 16, 32, 60, 60]) {
        yield* TestClock.adjust(Duration.seconds(wait))
        steps.push(yield* attempt("bad"))
        steps.push(yield* attempt("bad"))
      }
      yield* TestClock.adjust(Duration.minutes(1))
      steps.push(yield* attempt("mag-key"))
      steps.push(yield* attempt("bad"))
      return steps
    }))
    expect(timeline).toEqual([
      "wrong", "wrong", "wrong", "wrong", "wrong", "wait 1", "other SignInRefused",
      "wrong", "wait 2", "wrong", "wait 4", "wrong", "wait 8", "wrong", "wait 16",
      "wrong", "wait 32", "wrong", "wait 60", "wrong", "wait 60", "wrong", "wait 60",
      "ok", "wrong",
    ])
  })

  it("keeps a session alive while used and expires it after thirty idle days", async () => {
    const results = await run(Effect.gen(function* () {
      const remote = yield* makeRemoteAccess(network, memorySessionStore, true)
      const token = yield* remote.signIn("10.0.0.2", "mag-key")
      const steps = [yield* remote.access(remoteRequest(token)), yield* remote.access(remoteRequest())]
      yield* TestClock.adjust(Duration.days(29))
      steps.push(yield* remote.access(remoteRequest(token)))
      yield* TestClock.adjust(Duration.days(29))
      steps.push(yield* remote.access(remoteRequest(token)))
      yield* TestClock.adjust(Duration.sum(SESSION_IDLE_LIMIT, Duration.seconds(1)))
      steps.push(yield* remote.access(remoteRequest(token)))
      return steps
    }))
    expect(results).toEqual(["SignedIn", "Anonymous", "SignedIn", "SignedIn", "Anonymous"])
  })

  it("signs out one session and a new process knows none", async () => {
    const results = await run(Effect.gen(function* () {
      const remote = yield* makeRemoteAccess(network, memorySessionStore, true)
      const first = yield* remote.signIn("10.0.0.2", "mag-key")
      const second = yield* remote.signIn("10.0.0.2", "mag-key")
      yield* remote.signOut(Option.some(first))
      const restarted = yield* makeRemoteAccess(network, memorySessionStore, true)
      return [yield* remote.access(remoteRequest(first)), yield* remote.access(remoteRequest(second)), yield* restarted.access(remoteRequest(second))]
    }))
    expect(results).toEqual(["Anonymous", "SignedIn", "Anonymous"])
  })

  it("keeps sessions across a restart until the key changes, and never stores the token", async () => {
    const path = join(mkdtempSync(join(tmpdir(), "magnitude-remote-sessions-")), "remote-sessions.json")
    const results = await Effect.runPromise(Effect.gen(function* () {
      const remote = yield* makeRemoteAccess(network, yield* fileSessionStore(path, network), true)
      const kept = yield* remote.signIn("10.0.0.2", "mag-key")
      const ended = yield* remote.signIn("10.0.0.2", "mag-key")
      yield* remote.signOut(Option.some(ended))
      const restarted = yield* makeRemoteAccess(network, yield* fileSessionStore(path, network), true)
      const rekeyed = { ...network, apiKey: Option.some("mag-new") }
      const afterNewKey = yield* makeRemoteAccess(rekeyed, yield* fileSessionStore(path, rekeyed), true)
      return { kept, steps: [yield* restarted.access(remoteRequest(kept)), yield* restarted.access(remoteRequest(ended)), yield* afterNewKey.access(remoteRequest(kept))] }
    }).pipe(Effect.provide(BunFileSystem.layer)))
    expect(results.steps).toEqual(["SignedIn", "Anonymous", "Anonymous"])
    expect(readFileSync(path, "utf8")).not.toContain(results.kept)
  })

  it("treats loopback callers as local without a session", async () => {
    const access = await run(Effect.flatMap(makeRemoteAccess(network), remote => remote.access({ remoteAddress: Option.some("127.0.0.1"), cookies: {} } as never)))
    expect(access).toBe("Local")
  })
})
