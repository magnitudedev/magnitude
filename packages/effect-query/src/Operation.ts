import * as Context from "effect/Context"
import * as Effect from "effect/Effect"
import * as Schema from "effect/Schema"
import * as Stream from "effect/Stream"
import * as Key from "./Key.js"
import type { QueryKey } from "./Model.js"

export type Kind = "query" | "mutation" | "subscription" | "queryFromStream"
export type PayloadInput = Schema.Schema.AnyNoContext | Schema.Struct.Fields

/** @internal Runtime declaration carried directly by Effect Query primitives. */
export const TypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Operation")
/** @internal Type-level failure of a supplied operation implementation. */
export const ImplementationErrorTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Operation/ImplementationError")

export interface Implementations<ImplementationError> {
  readonly [ImplementationErrorTypeId]: (_: ImplementationError) => ImplementationError
}

export type ImplementationError<Value> = Value extends Implementations<infer Error> ? Error : never

export interface ImplementationService<ImplementationError> {
  readonly execute: (operation: Any, payload: unknown) => Effect.Effect<unknown, unknown | ImplementationError>
  readonly stream: (operation: Any, payload: unknown) => Stream.Stream<unknown, unknown | ImplementationError>
}

const ImplementationsTag = Context.GenericTag<ImplementationService<unknown>>(
  "@magnitudedev/effect-query/Operation/Implementations",
)

/** @internal The shared DI slot adapters satisfy with local, RPC, or test implementations. */
export const implementationsTag = <Error>(): Context.Tag<Implementations<Error>, ImplementationService<Error>> =>
  ImplementationsTag as unknown as Context.Tag<Implementations<Error>, ImplementationService<Error>>

export interface Declaration<
  Name extends string,
  OperationKind extends Kind,
  Payload extends PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Policy extends object,
> {
  readonly name: Name
  readonly kind: OperationKind
  readonly payload: Payload
  readonly success: Success
  readonly error: Error
  readonly annotations: Context.Context<never>
  readonly policy: Policy
}

export interface Declared<
  Name extends string,
  OperationKind extends Kind,
  Payload extends PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Policy extends object,
> {
  readonly [TypeId]: Declaration<Name, OperationKind, Payload, Success, Error, Policy>
}

export type Any = Declared<
  string,
  Kind,
  PayloadInput,
  Schema.Schema.Any,
  Schema.Schema.All,
  object
>

export type Name<Value> = Value extends Declared<infer ValueName, Kind, PayloadInput, Schema.Schema.Any, Schema.Schema.All, object>
  ? ValueName
  : never
export type OperationKind<Value> = Value extends Declared<string, infer ValueKind, PayloadInput, Schema.Schema.Any, Schema.Schema.All, object>
  ? ValueKind
  : never
export type Payload<Value> = Value extends Declared<string, Kind, infer ValuePayload, Schema.Schema.Any, Schema.Schema.All, object>
  ? ValuePayload
  : never
export type SuccessSchema<Value> = Value extends Declared<string, Kind, PayloadInput, infer ValueSuccess, Schema.Schema.All, object>
  ? ValueSuccess
  : never
export type ErrorSchema<Value> = Value extends Declared<string, Kind, PayloadInput, Schema.Schema.Any, infer ValueError, object>
  ? ValueError
  : never
export type Policy<Value> = Value extends Declared<string, Kind, PayloadInput, Schema.Schema.Any, Schema.Schema.All, infer ValuePolicy>
  ? ValuePolicy
  : never

export interface Shape<
  Payload extends PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Policy extends object = {},
> {
  readonly payload?: Payload
  readonly success?: Success
  readonly error?: Error
  readonly annotations?: Context.Context<never>
  readonly policy?: Policy
}

export type PayloadConstructor<Payload extends PayloadInput> =
  Payload extends { readonly fields: infer Fields extends Schema.Struct.Fields }
    ? Schema.Simplify<Schema.Struct.Constructor<Fields>>
    : Payload extends Schema.Struct.Fields
    ? Schema.Simplify<Schema.Struct.Constructor<Payload>>
    : Payload extends Schema.Schema.Any ? Schema.Schema.Type<Payload> : never

const defaultPayload = Schema.Void
const defaultSuccess = Schema.Void
const defaultError = Schema.Never

export const attach = <
  Definition extends object,
  const Name extends string,
  const OperationKind extends Kind,
  Payload extends PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Policy extends object,
>(
  definition: Definition,
  name: Name,
  kind: OperationKind,
  options: Shape<Payload, Success, Error, Policy>,
): Definition & Declared<Name, OperationKind, Payload, Success, Error, Policy> =>
  Object.assign(definition, {
    [TypeId]: {
      name,
      kind,
      payload: options.payload ?? defaultPayload,
      success: options.success ?? defaultSuccess,
      error: options.error ?? defaultError,
      annotations: options.annotations ?? Context.empty(),
      policy: options.policy ?? {},
    },
  }) as never

export const isDeclared = (value: unknown): value is Any =>
  typeof value === "object" && value !== null && TypeId in value

export const declaration = <Value extends Any>(value: Value): Value[typeof TypeId] => value[TypeId]

interface Constructible {
  readonly make: (input: unknown) => unknown
}

const isConstructible = (schema: object): schema is Constructible =>
  "make" in schema && typeof schema.make === "function"

export const payloadKey = (payload: PayloadInput) => (input: unknown): QueryKey => {
  const schema = payload as object
  return Key.canonical(isConstructible(schema) ? schema.make(input) : input)
}

/** @internal Execute through whichever implementation layer the client supplied. */
export const execute = <Value extends Any>(value: Value, input: unknown): Effect.Effect<unknown, unknown, Implementations<unknown>> =>
  Effect.flatMap(implementationsTag<unknown>(), (implementations) => implementations.execute(value, input))

/** @internal Stream through whichever implementation layer the client supplied. */
export const stream = <Value extends Any>(value: Value, input: unknown): Stream.Stream<unknown, unknown, Implementations<unknown>> =>
  Stream.unwrap(Effect.map(implementationsTag<unknown>(), (implementations) => implementations.stream(value, input)))
