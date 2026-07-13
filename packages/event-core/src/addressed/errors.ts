import { Data } from 'effect'

export class AddressedStoreError extends Data.TaggedError('AddressedStoreError')<{
  readonly operation: 'load' | 'flush' | 'stat'
  readonly namespace: string
  readonly address: string
  readonly cause: unknown
}> {}

export class AddressedCodecError extends Data.TaggedError('AddressedCodecError')<{
  readonly operation: 'encode' | 'decode'
  readonly namespace: string
  readonly address: string
  readonly cause: unknown
}> {}

export class AddressedCollectionError extends Data.TaggedError('AddressedCollectionError')<{
  readonly collection: string
  readonly address?: string
  readonly operation: string
  readonly reason: string
}> {}

export type AddressedError =
  | AddressedStoreError
  | AddressedCodecError
  | AddressedCollectionError
