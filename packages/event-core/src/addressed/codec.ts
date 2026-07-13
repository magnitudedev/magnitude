import { Effect, Schema } from 'effect'
import { AddressedCodecError } from './errors'

export interface AddressedEntryCodec<Value> {
  readonly encode: (
    value: Value,
    context: { readonly namespace: string; readonly address: string }
  ) => Effect.Effect<unknown, AddressedCodecError>

  readonly decode: (
    encoded: unknown,
    context: { readonly namespace: string; readonly address: string }
  ) => Effect.Effect<Value, AddressedCodecError>
}

export const makeSchemaCodec = <S extends Schema.Schema.AnyNoContext>(
  schema: S
): AddressedEntryCodec<Schema.Schema.Type<S>> => ({
  encode: (value, context) =>
    Schema.encode(schema)(value).pipe(
      Effect.mapError((cause) =>
        new AddressedCodecError({
          operation: 'encode',
          namespace: context.namespace,
          address: context.address,
          cause
        })
      )
    ),

  decode: (encoded, context) =>
    Schema.decodeUnknown(schema)(encoded).pipe(
      Effect.mapError((cause) =>
        new AddressedCodecError({
          operation: 'decode',
          namespace: context.namespace,
          address: context.address,
          cause
        })
      )
    )
})
