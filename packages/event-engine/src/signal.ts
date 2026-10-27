import { Data, type DateTime, type Effect, type Option, type Scope, type Stream } from "effect"
import type { AddressRef } from "./address"
import type { ProjectionDefinitionBase } from "./projection"

const SignalTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Signal")
const SignalFamilyTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Signal/Family")
const SignalPayloadTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Signal/Payload")

export class SignalCascadeExceeded extends Data.TaggedError("SignalCascadeExceeded")<{
  readonly limit: number
}> {}

export interface SignalDefinition<Payload> {
  readonly [SignalTypeId]: typeof SignalTypeId
  readonly [SignalFamilyTypeId]: object
  readonly [SignalPayloadTypeId]: Payload
  readonly name: string
}

export type SignalPayload<Definition extends SignalDefinition<unknown>> =
  Definition[typeof SignalPayloadTypeId]

export interface SignalOccurrence<Payload> {
  readonly signal: SignalDefinition<Payload>
  readonly payload: Payload
  readonly source: {
    readonly projection: ProjectionDefinitionBase
    readonly address: Option.Option<AddressRef>
  }
  readonly timestamp: DateTime.Utc
}

export type OccurrenceOf<Definition extends SignalDefinition<unknown>> =
  Definition extends SignalDefinition<infer Payload> ? SignalOccurrence<Payload> : never

export type SignalEmitter<Signals extends readonly SignalDefinition<unknown>[]> =
  <Definition extends Signals[number]>(
    signal: Definition,
    payload: SignalPayload<Definition>,
  ) => void

export interface SignalObserver {
  readonly stream: <Payload>(
    signal: SignalDefinition<Payload>,
  ) => Stream.Stream<SignalOccurrence<Payload>>
  readonly subscribe: <Payload>(
    signal: SignalDefinition<Payload>,
  ) => Effect.Effect<Stream.Stream<SignalOccurrence<Payload>>, never, Scope.Scope>
}

export const SignalInternal = {
  belongsTo: (signal: SignalDefinition<unknown>, family: object): boolean =>
    signal[SignalFamilyTypeId] === family,
}

export const makeSignalBuilder = (family: object) => ({
  define: <Payload>(config: { readonly name: string }): SignalDefinition<Payload> => ({
    [SignalTypeId]: SignalTypeId,
    [SignalFamilyTypeId]: family,
    [SignalPayloadTypeId]: undefined as Payload,
    name: config.name,
  }),
})
