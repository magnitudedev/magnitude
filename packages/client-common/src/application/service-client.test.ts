import { describe, expect, it } from "vitest"
import { HttpClient, HttpClientError, HttpClientResponse } from "@effect/platform"
import { Context, Effect, Exit, Fiber, Layer, Option, Schema, Stream, SubscriptionRef } from "effect"
import { MagnitudeClient, MagnitudeHealthResponseSchema, MAGNITUDE_RPC_VERSION, type ConnectionState } from "@magnitudedev/sdk"
import { Failed, Ready, Starting, type ApplicationSnapshot, type OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import { applicationClientLayer } from "./service-client"
import type { ApplicationHost } from "./host"

const ORIGIN = "http://127.0.0.1:4000"
const answer = {
  service: "magnitude-acn", version: "0.0.1", revision: 1, rpcVersion: MAGNITUDE_RPC_VERSION, id: "first", pid: 123, state: { _tag: "Ready" },
}
const health = Schema.decodeUnknownSync(MagnitudeHealthResponseSchema)(answer)

/** A service that refuses connections until it listens, then answers ready. */
const service = () => {
  let listening = false
  const http = HttpClient.make((request) => Effect.suspend(() => !listening
    ? Effect.fail(new HttpClientError.RequestError({ request, reason: "Transport" }))
    : Effect.succeed(HttpClientResponse.fromWeb(request, Response.json(answer)))))
  return { http, listen: () => { listening = true } }
}

const snapshot = (state: OwnedServiceState): ApplicationSnapshot => ({
  version: 1, pid: 1, endpoint: ORIGIN, service: state, owner: { _tag: "Desktop", tray: { _tag: "Registered" } },
})

const host = (application: Option.Option<Stream.Stream<ApplicationSnapshot>>): ApplicationHost => ({
  window: Option.none(),
  appearance: { read: Effect.succeed("system"), save: () => Effect.void },
  shell: Option.map(application, (application) => ({
    actions: Stream.never,
    presentModel: () => Effect.void,
    resolveQuitFailure: () => Effect.void,
    retryService: Effect.void,
    chooseOpenPath: () => Effect.succeedNone,
    application,
  })),
})

/** Connects as the window would, recording every connection state, while `drive` plays the service's startup. */
const connect = (
  application: Option.Option<Stream.Stream<ApplicationSnapshot>>,
  drive: (listen: () => void) => Effect.Effect<void>,
) => Effect.gen(function* () {
  const fake = service()
  const client = yield* Layer.build(applicationClientLayer({ origin: ORIGIN, host: host(application), reconnect: "Automatic" }).pipe(
    Layer.provide(Layer.succeed(HttpClient.HttpClient, fake.http))))
  const sdk = Context.get(client, MagnitudeClient)
  const states: ConnectionState["_tag"][] = []
  yield* sdk.connection.changes.pipe(Stream.runForEach((state) => Effect.sync(() => states.push(state._tag))), Effect.forkScoped)
  const attempt = yield* Effect.fork(sdk.connection.connect)
  yield* Effect.sleep("50 millis")
  yield* drive(fake.listen)
  return { exit: yield* Fiber.await(attempt), states }
}).pipe(Effect.scoped, Effect.runPromise)

describe("a window's service connection", () => {
  it("waits while its owner starts the service, without failing", async () => {
    const owner = Effect.runSync(SubscriptionRef.make(snapshot(new Starting({ attempt: 0, health: Option.none() }))))
    const { exit, states } = await connect(Option.some(owner.changes), (listen) => Effect.sync(listen).pipe(
      Effect.zipRight(SubscriptionRef.set(owner, snapshot(new Ready({ health })))),
    ))
    expect(exit).toEqual(Exit.void)
    expect(states).not.toContain("Failed")
  })

  it("fails with the owner's message when the service could not start", async () => {
    const owner = Effect.runSync(SubscriptionRef.make(snapshot(new Starting({ attempt: 0, health: Option.none() }))))
    const { exit } = await connect(Option.some(owner.changes), () =>
      SubscriptionRef.set(owner, snapshot(new Failed({ message: "Magnitude service could not start." }))))
    expect(exit).toEqual(Exit.fail(expect.objectContaining({ _tag: "ServiceStartFailed", message: "Magnitude service could not start." })))
  })

  it("reports a browser's unreachable service at once", async () => {
    const { exit } = await connect(Option.none(), () => Effect.void)
    expect(exit).toEqual(Exit.fail(expect.objectContaining({ _tag: "ServiceUnavailable" })))
  })
})
