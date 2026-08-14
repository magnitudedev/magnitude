import { RpcServer } from "@effect/rpc"
import type {
  FromClientEncoded,
  FromServerEncoded,
} from "@effect/rpc/RpcMessage"
import {
  AcnSubscriptionMetadataTag,
  MagnitudeRpcs,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, Exit, Layer, Mailbox, Option, Ref, Schema, Stream } from "effect"
import { AcnSubscriptions } from "./acn-subscriptions"

class RawAcnRpcProtocol extends Context.Tag("RawAcnRpcProtocol")<
  RawAcnRpcProtocol,
  RpcServer.Protocol["Type"]
>() {}

const SessionScopedPayload = Schema.Struct({ sessionId: Schema.String })
const decodeSessionScopedPayload = Schema.decodeUnknown(SessionScopedPayload)

const subscriptionMetadata = (tag: string) => {
  const rpc = MagnitudeRpcs.requests.get(tag)
  return rpc
    ? Context.getOption(rpc.annotations, AcnSubscriptionMetadataTag)
    : Option.none()
}

const requestSessionId = (
  request: Extract<FromClientEncoded, { readonly _tag: "Request" }>
) =>
  decodeSessionScopedPayload(request.payload).pipe(
    Effect.map((payload) => payload.sessionId)
  )

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

export const makeAcnSubscriptionProtocol = (
  protocol: RpcServer.Protocol["Type"]
): Effect.Effect<RpcServer.Protocol["Type"], never, AcnSubscriptions> =>
  Effect.gen(function* () {
    const subscriptions = yield* AcnSubscriptions
    interface FinalizerEntry {
      readonly token: object
      readonly unregister: Effect.Effect<void>
    }
    const finalizers = yield* Ref.make(
      new Map<number, ReadonlyMap<string, FinalizerEntry>>()
    )
    // A poison entry exists only while at least one forwarded request from the
    // closed transport still awaits its authoritative terminal Exit.
    const closedClients = yield* Ref.make<ReadonlySet<number>>(new Set())
    const disconnects = yield* Mailbox.make<number>()

    const remove = (clientId: number, requestId: string) =>
      Ref.modify(finalizers, (all) => {
        const client = all.get(clientId)
        const unregister = client?.get(requestId)?.unregister ?? Effect.void
        if (!client?.has(requestId)) return [{ unregister, clientEmpty: false }, all] as const

        const nextClient = new Map(client)
        nextClient.delete(requestId)
        const next = new Map(all)
        if (nextClient.size === 0) next.delete(clientId)
        else next.set(clientId, nextClient)
        return [{ unregister, clientEmpty: nextClient.size === 0 }, next] as const
      }).pipe(
        Effect.flatMap(({ unregister, clientEmpty }) => unregister.pipe(
          Effect.zipRight(clientEmpty
            ? Ref.update(closedClients, (closed) => {
                const next = new Set(closed)
                next.delete(clientId)
                return next
              })
            : Effect.void),
        )),
      )

    const removeClient = (clientId: number) =>
      Effect.gen(function* () {
        const entries = yield* Ref.modify(finalizers, (all) => {
          const client = all.get(clientId)
          if (!client) return [[] as readonly FinalizerEntry[], all] as const
          const next = new Map(all)
          next.delete(clientId)
          return [Array.from(client.values()), next] as const
        })
        yield* Effect.forEach(entries, (entry) => entry.unregister, {
          discard: true,
          concurrency: "unbounded",
        })
        yield* Ref.update(closedClients, (closed) => {
          const next = new Set(closed)
          next.delete(clientId)
          return next
        })
      })

    const unregister = (clientId: number, requestId: string) =>
      Ref.get(finalizers).pipe(
        Effect.flatMap((all) =>
          all.get(clientId)?.get(requestId)?.unregister ?? Effect.void
        ),
      )

    const poisonClient = (clientId: number) =>
      Effect.gen(function* () {
        yield* Ref.update(closedClients, (closed) => new Set(closed).add(clientId))
        const entries = yield* Ref.modify(finalizers, (all) => {
          const client = all.get(clientId)
          if (!client) return [[] as readonly FinalizerEntry[], all] as const
          const disarmed = new Map(
            Array.from(client, ([requestId, entry]) => [
              requestId,
              { ...entry, unregister: Effect.void },
            ]),
          )
          return [Array.from(client.values()), new Map(all).set(clientId, disarmed)] as const
        })
        yield* Effect.forEach(entries, (entry) => entry.unregister, {
          discard: true,
          concurrency: "unbounded",
        })
        yield* protocol.end(clientId)
      })

    const register = (
      clientId: number,
      request: Extract<FromClientEncoded, { readonly _tag: "Request" }>
    ) =>
      Effect.uninterruptible(
        Effect.gen(function* () {
          const token = {}
          const reserved = yield* Ref.modify(finalizers, (all) => {
            const client = new Map(all.get(clientId) ?? [])
            if (client.has(request.id)) return [false, all] as const
            client.set(request.id, { token, unregister: Effect.void })
            return [true, new Map(all).set(clientId, client)] as const
          })
          if (!reserved) {
            yield* poisonClient(clientId)
            return false
          }

          const metadata = subscriptionMetadata(request.tag)
          if (Option.isNone(metadata)) return true
          const sessionId = metadata.value.scope === "session"
            ? yield* requestSessionId(request).pipe(Effect.option)
            : Option.none<string>()
          // A malformed payload is still owned until the RPC decoder emits Exit,
          // but it cannot register a transport subscription.
          if (metadata.value.scope === "session" && Option.isNone(sessionId)) return true

          const registered = yield* Effect.exit(subscriptions.register({
            clientId,
            requestId: request.id,
            ...Option.match(sessionId, {
              onNone: () => ({}),
              onSome: (value) => ({ sessionId: value }),
            }),
            emit: (control) =>
              protocol.send(clientId, {
                _tag: "Chunk",
                requestId: request.id,
                values: [control],
              }),
            close: protocol.end(clientId),
          }))
          if (Exit.isFailure(registered)) {
            yield* remove(clientId, request.id)
            yield* protocol.end(clientId)
            return false
          }
          const handle = registered.value
          const installed = yield* Ref.modify(finalizers, (all) => {
            const client = new Map(all.get(clientId) ?? [])
            const current = client.get(request.id)
            if (current?.token !== token) return [false, all] as const
            client.set(request.id, { token, unregister: handle.unregister })
            return [true, new Map(all).set(clientId, client)] as const
          })
          if (!installed) yield* handle.unregister
          return installed
        }),
      )

    const onRequest = (clientId: number, request: FromClientEncoded) =>
      Ref.get(closedClients).pipe(
        Effect.flatMap((closed) => {
          if (closed.has(clientId)) return Effect.succeed(false)
          switch (request._tag) {
            case "Request":
              return register(clientId, request)
            case "Interrupt":
              return unregister(clientId, request.requestId).pipe(Effect.as(true))
            case "Ack":
            case "Eof":
            case "Ping":
              return Effect.succeed(true)
          }
        }),
      )

    const send = (clientId: number, response: FromServerEncoded) =>
      response._tag === "Exit"
        ? protocol
            .send(clientId, response)
            .pipe(Effect.ensuring(remove(clientId, response.requestId)))
        : protocol.send(clientId, response)

    return RpcServer.Protocol.of({
      ...protocol,
      disconnects,
      run: (writeRequest) =>
        Effect.raceFirst(
          protocol.run((clientId, request) =>
            onRequest(clientId, request).pipe(
              Effect.flatMap((shouldForward) => {
                if (!shouldForward) return Effect.void
                const unknownRequest = request._tag === "Request"
                  && !MagnitudeRpcs.requests.has(request.tag)
                return writeRequest(clientId, request).pipe(
                  Effect.ensuring(
                    unknownRequest ? remove(clientId, request.id) : Effect.void,
                  ),
                )
              }),
            )
          ),
          Mailbox.toStream(protocol.disconnects).pipe(
            Stream.runForEach((clientId) =>
              removeClient(clientId).pipe(
                Effect.catchAllCause(() => Effect.void),
                Effect.zipRight(disconnects.offer(clientId)),
                Effect.asVoid,
              ),
            ),
            Effect.zipRight(Effect.never),
          ),
        ),
      send,
    })
  })
