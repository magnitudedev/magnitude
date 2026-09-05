import { RpcSchema, RpcServer } from "@effect/rpc"
import type {
  FromClientEncoded,
  FromServerEncoded,
} from "@effect/rpc/RpcMessage"
import { AcnRpcGroup } from "@magnitudedev/acn-protocol/boundary"
import { Array as Arr, Context, Effect, Layer, Ref } from "effect"
import { AcnSubscriptions } from "./acn-subscriptions"

class RawAcnRpcProtocol extends Context.Tag("RawAcnRpcProtocol")<
  RawAcnRpcProtocol,
  RpcServer.Protocol["Type"]
>() {}

export const acnSubscriptionProtocolLayer = <Error, Requirements>(
  rawProtocol: Layer.Layer<RpcServer.Protocol, Error, Requirements>
): Layer.Layer<RpcServer.Protocol, Error, AcnSubscriptions | Requirements> => {
  const raw = Layer.effect(RawAcnRpcProtocol, RpcServer.Protocol).pipe(
    Layer.provide(rawProtocol)
  )

  return Layer.effect(
    RpcServer.Protocol,
    Effect.gen(function* () {
      const protocol = yield* RawAcnRpcProtocol
      return yield* makeAcnSubscriptionProtocol(protocol)
    })
  ).pipe(Layer.provide(raw))
}

/**
 * Decorates the RPC server protocol with the ACN subscription wire protocol.
 *
 * Every stream Rpc of the ACN group is a subscription: its encoded chunk
 * values are wrapped in `payload` frames, keepalives are interleaved, and
 * shutdown emits the terminal control. Handlers and clients see domain values.
 */
export const makeAcnSubscriptionProtocol = (
  protocol: RpcServer.Protocol["Type"]
): Effect.Effect<RpcServer.Protocol["Type"], never, AcnSubscriptions> =>
  Effect.gen(function* () {
    const subscriptions = yield* AcnSubscriptions
    const finalizers = yield* Ref.make(
      new Map<number, ReadonlyMap<string, Effect.Effect<void>>>()
    )
    const subscriptionRequests = yield* Ref.make(new Map<number, ReadonlySet<string>>())

    const track = (clientId: number, requestId: string, tracked: boolean) =>
      Ref.update(subscriptionRequests, (all) => {
        const client = new Set(all.get(clientId) ?? [])
        if (tracked) client.add(requestId)
        else client.delete(requestId)
        const next = new Map(all)
        if (client.size === 0) next.delete(clientId)
        else next.set(clientId, client)
        return next
      })

    const isSubscriptionRequest = (clientId: number, requestId: string) =>
      Ref.get(subscriptionRequests).pipe(
        Effect.map((all) => all.get(clientId)?.has(requestId) === true),
      )

    const remove = (clientId: number, requestId: string) =>
      Ref.modify(finalizers, (all) => {
        const client = all.get(clientId)
        const finalizer = client?.get(requestId) ?? Effect.void
        if (!client?.has(requestId)) return [finalizer, all] as const

        const nextClient = new Map(client)
        nextClient.delete(requestId)
        const next = new Map(all)
        if (nextClient.size === 0) next.delete(clientId)
        else next.set(clientId, nextClient)
        return [finalizer, next] as const
      }).pipe(Effect.flatten, Effect.zipRight(track(clientId, requestId, false)))

    const register = (
      clientId: number,
      request: Extract<FromClientEncoded, { readonly _tag: "Request" }>
    ) =>
      Effect.gen(function* () {
        if (!(AcnRpcGroup.requests.get(request.tag) && RpcSchema.isStreamSchema(AcnRpcGroup.requests.get(request.tag)!.successSchema))) return
        yield* track(clientId, request.id, true)
        const handle = yield* subscriptions.register({
          clientId,
          requestId: request.id,
          emit: (control) =>
            protocol.send(clientId, {
              _tag: "Chunk",
              requestId: request.id,
              values: [control],
            }),
          close: protocol.end(clientId),
        })
        const previous = yield* Ref.modify(finalizers, (all) => {
          const client = new Map(all.get(clientId) ?? [])
          const prior = client.get(request.id) ?? Effect.void
          client.set(request.id, handle.unregister)
          return [prior, new Map(all).set(clientId, client)] as const
        })
        yield* previous
      })

    const onRequest = (clientId: number, request: FromClientEncoded) => {
      switch (request._tag) {
        case "Request":
          return register(clientId, request)
        case "Interrupt":
          return remove(clientId, request.requestId)
        case "Ack":
        case "Eof":
        case "Ping":
          return Effect.void
      }
    }

    const send = (clientId: number, response: FromServerEncoded) => {
      switch (response._tag) {
        case "Chunk":
          return isSubscriptionRequest(clientId, response.requestId).pipe(
            Effect.flatMap((subscription) => protocol.send(clientId, subscription
              ? {
                  ...response,
                  values: Arr.map(response.values, (value) => ({ _tag: "payload", payload: value })),
                }
              : response)),
          )
        case "Exit":
          return protocol
            .send(clientId, response)
            .pipe(Effect.ensuring(remove(clientId, response.requestId)))
        default:
          return protocol.send(clientId, response)
      }
    }

    return RpcServer.Protocol.of({
      ...protocol,
      run: (writeRequest) =>
        protocol.run((clientId, request) =>
          onRequest(clientId, request).pipe(
            Effect.zipRight(writeRequest(clientId, request))
          )
        ),
      send,
    })
  })
