import { RpcServer } from "@effect/rpc"
import type { FromClientEncoded, FromServerEncoded } from "@effect/rpc/RpcMessage"
import {
  AcnSubscriptionMetadataTag,
  MagnitudeRpcs,
} from "@magnitudedev/protocol"
import { Context, Effect, Layer, Option, Ref, Schema } from "effect"
import { AcnSubscriptions } from "./acn-subscriptions"

class RawAcnRpcProtocol extends Context.Tag("RawAcnRpcProtocol")<
  RawAcnRpcProtocol,
  RpcServer.Protocol["Type"]
>() {}

const SessionScopedPayload = Schema.Struct({ sessionId: Schema.String })
const decodeSessionScopedPayload = Schema.decodeUnknown(SessionScopedPayload)

const subscriptionMetadata = (tag: string) => {
  const rpc = MagnitudeRpcs.requests.get(tag)
  return rpc ? Context.getOption(rpc.annotations, AcnSubscriptionMetadataTag) : Option.none()
}

const requestSessionId = (request: Extract<FromClientEncoded, { readonly _tag: "Request" }>) =>
  decodeSessionScopedPayload(request.payload).pipe(Effect.map((payload) => payload.sessionId))

export const acnSubscriptionProtocolLayer = <Error, Requirements>(
  rawProtocol: Layer.Layer<RpcServer.Protocol, Error, Requirements>,
): Layer.Layer<RpcServer.Protocol, Error, AcnSubscriptions | Requirements> => {
  const raw = Layer.effect(RawAcnRpcProtocol, RpcServer.Protocol).pipe(
    Layer.provide(rawProtocol),
  )

  return Layer.effect(
    RpcServer.Protocol,
    Effect.gen(function* () {
      const protocol = yield* RawAcnRpcProtocol
      const subscriptions = yield* AcnSubscriptions
      const finalizers = yield* Ref.make(
        new Map<number, ReadonlyMap<string, Effect.Effect<void>>>(),
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
        }).pipe(Effect.flatten)

      const register = (
        clientId: number,
        request: Extract<FromClientEncoded, { readonly _tag: "Request" }>,
      ) =>
        Effect.gen(function* () {
          const metadata = subscriptionMetadata(request.tag)
          if (Option.isNone(metadata)) return
          const sessionId = metadata.value.scope === "session"
            ? yield* requestSessionId(request)
            : undefined
          const handle = yield* subscriptions.register({
            clientId,
            requestId: request.id,
            ...(sessionId === undefined ? {} : { sessionId }),
            emit: (control) =>
              protocol.send(clientId, {
                _tag: "Chunk",
                requestId: request.id,
                values: [control],
              }),
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

      const send = (clientId: number, response: FromServerEncoded) =>
        response._tag === "Exit"
          ? protocol.send(clientId, response).pipe(
              Effect.ensuring(remove(clientId, response.requestId)),
            )
          : protocol.send(clientId, response)

      return RpcServer.Protocol.of({
        ...protocol,
        run: (writeRequest) =>
          protocol.run((clientId, request) =>
            onRequest(clientId, request).pipe(
              Effect.catchAll(() => Effect.void),
              Effect.zipRight(writeRequest(clientId, request)),
            ),
          ),
        send,
      })
    }),
  ).pipe(Layer.provide(raw))
}
