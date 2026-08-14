import { RpcServer } from "@effect/rpc"
import type { FromClientEncoded, FromServerEncoded } from "@effect/rpc/RpcMessage"
import { Effect, Mailbox, Ref } from "effect"
import { describe, expect, it } from "vitest"
import { makeAcnSubscriptionProtocol } from "./acn-subscription-protocol"
import {
  AcnSubscriptions,
  type AcnSubscriptionRegistration,
  type AcnSubscriptionsApi,
} from "./acn-subscriptions"

const streamRequest: Extract<FromClientEncoded, { readonly _tag: "Request" }> = {
  _tag: "Request",
  id: "duplicate-request",
  tag: "StreamDisplayView",
  payload: { sessionId: "session-1" },
  headers: [],
}

const runRequests = async (
  requests: readonly FromClientEncoded[],
  responsesAfterRequests: readonly FromServerEncoded[] = [],
  requestsAfterResponses: readonly FromClientEncoded[] = [],
  disconnectAfterRequests = false,
) => {
  let active: AcnSubscriptionRegistration | undefined
  const subscriptions: AcnSubscriptionsApi = {
    register: (registration) => Effect.sync(() => {
      active = registration
      return {
        unregister: Effect.sync(() => {
          if (active === registration) active = undefined
        }),
      }
    }),
    suspendSession: (sessionId) => active?.sessionId === sessionId
      ? active.emit({ _tag: "suspended", reason: "session-offloaded" })
      : Effect.void,
    terminate: Effect.void,
  }
  const program = Effect.gen(function*() {
    const forwarded = yield* Ref.make<FromClientEncoded[]>([])
    const sent = yield* Ref.make<FromServerEncoded[]>([])
    const ended = yield* Ref.make(0)
    const disconnects = yield* Mailbox.make<number>()
    let wrapped!: RpcServer.Protocol["Type"]
    const raw = {
      run: (accept: (clientId: number, request: FromClientEncoded) => Effect.Effect<void>) =>
        Effect.forEach(requests, (request) => accept(7, request), { discard: true }).pipe(
          Effect.zipRight(
            disconnectAfterRequests
              ? disconnects.offer(7).pipe(
                  Effect.zipRight(
                    Effect.suspend(function waitForDisconnect(): Effect.Effect<void> {
                      return active === undefined
                        ? Effect.void
                        : Effect.yieldNow().pipe(Effect.zipRight(Effect.suspend(waitForDisconnect)))
                    }),
                  ),
                )
              : Effect.void,
          ),
          Effect.zipRight(
            Effect.forEach(
              responsesAfterRequests,
              (response) => wrapped.send(7, response),
              { discard: true },
            ),
          ),
          Effect.zipRight(
            Effect.forEach(requestsAfterResponses, (request) => accept(7, request), {
              discard: true,
            }),
          ),
        ),
      disconnects,
      send: (_clientId: number, response: FromServerEncoded) =>
        Ref.update(sent, (responses) => [...responses, response]),
      end: (_clientId: number) => Ref.update(ended, (count) => count + 1),
    } as unknown as RpcServer.Protocol["Type"]
    wrapped = yield* makeAcnSubscriptionProtocol(raw)
    yield* wrapped.run((_clientId, request) =>
      Ref.update(forwarded, (all) => [...all, request])
    ) as unknown as Effect.Effect<void>
    const disconnected = disconnectAfterRequests
      ? (yield* wrapped.disconnects.take) === 7
      : false

    yield* subscriptions.suspendSession("session-1")

    return {
      forwarded: yield* Ref.get(forwarded),
      sent: yield* Ref.get(sent),
      ended: yield* Ref.get(ended),
      disconnected,
    }
  }).pipe(Effect.provideService(AcnSubscriptions, subscriptions))
  return Effect.runPromise(program)
}

describe("makeAcnSubscriptionProtocol", () => {
  it("fails closed instead of letting an overlapping request id replace its owner", async () => {
    const result = await runRequests([streamRequest, { ...streamRequest }])

    expect(result.forwarded).toEqual([streamRequest])
    expect(result.sent).toEqual([])
    expect(result.ended).toBe(1)
  })

  it("fails closed for duplicate ordinary RPC request ids", async () => {
    const request: Extract<FromClientEncoded, { readonly _tag: "Request" }> = {
      _tag: "Request",
      id: "ordinary-duplicate",
      tag: "GetSession",
      payload: { sessionId: "session-1" },
      headers: [],
    }
    const result = await runRequests([request, { ...request }])

    expect(result.forwarded).toEqual([request])
    expect(result.ended).toBe(1)
  })

  it("releases unknown RPC request ids after forwarding defects", async () => {
    const request: Extract<FromClientEncoded, { readonly _tag: "Request" }> = {
      _tag: "Request",
      id: "unknown-request",
      tag: "UnknownRpc",
      payload: {},
      headers: [],
    }
    const result = await runRequests([request, { ...request }])

    expect(result.forwarded).toEqual([request, request])
    expect(result.ended).toBe(0)
  })

  it("releases in-flight reservations when the transport disconnects", async () => {
    const result = await runRequests(
      [streamRequest],
      [],
      [{ ...streamRequest }],
      true,
    )

    expect(result.forwarded).toEqual([streamRequest, streamRequest])
    expect(result.ended).toBe(0)
    expect(result.disconnected).toBe(true)
  })

  it("keeps an interrupted request id reserved until its terminal response", async () => {
    const interrupt: FromClientEncoded = {
      _tag: "Interrupt",
      requestId: streamRequest.id,
    }
    const result = await runRequests([streamRequest, interrupt, { ...streamRequest }])

    expect(result.forwarded).toEqual([streamRequest, interrupt])
    expect(result.sent).toEqual([])
    expect(result.ended).toBe(1)
  })

  it("keeps an overlapped client closed through queued callbacks and predecessor Exit", async () => {
    const predecessorExit = {
      _tag: "Exit",
      requestId: streamRequest.id,
      exit: { _tag: "Success", value: undefined },
    } as unknown as FromServerEncoded
    const result = await runRequests(
      [streamRequest, { ...streamRequest }, { ...streamRequest }],
      [predecessorExit],
    )

    expect(result.forwarded).toEqual([streamRequest])
    expect(result.sent).toEqual([predecessorExit])
    expect(result.ended).toBe(1)
  })

  it("releases poison state after the predecessor's terminal Exit", async () => {
    const predecessorExit = {
      _tag: "Exit",
      requestId: streamRequest.id,
      exit: { _tag: "Success", value: undefined },
    } as unknown as FromServerEncoded
    const result = await runRequests(
      [streamRequest, { ...streamRequest }],
      [predecessorExit],
      [{ ...streamRequest }],
    )

    expect(result.forwarded).toEqual([streamRequest, streamRequest])
    expect(result.ended).toBe(1)
  })
})
