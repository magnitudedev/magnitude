import { Data, Effect, Equal, Hash, Option, Schema } from "effect"

const AddressTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Address")
const AddressSpaceTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/AddressSpace")
const AddressSpaceParentTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/AddressSpace/Parent")
const AddressSpaceKeyTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/AddressSpace/Key")
const AddressSpaceEncodedKeyTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/AddressSpace/EncodedKey")

export interface AddressRoot {
  readonly _tag: "AddressRoot"
}

export const AddressRoot: AddressRoot = Object.freeze({ _tag: "AddressRoot" })

export interface AddressDocumentSegment {
  readonly space: string
  readonly key: AddressEncodedValue
}

export interface AddressDocument {
  readonly segments: readonly AddressDocumentSegment[]
}

export type AddressEncodedValue =
  | null
  | boolean
  | number
  | string
  | readonly AddressEncodedValue[]
  | { readonly [key: string]: AddressEncodedValue }

export class AddressSegmentCountMismatch extends Data.TaggedError("AddressSegmentCountMismatch")<{
  readonly space: string
  readonly expected: number
  readonly actual: number
}> {}

export class AddressSpaceMismatch extends Data.TaggedError("AddressSpaceMismatch")<{
  readonly index: number
  readonly expected: string
  readonly actual: string | null
}> {}

export class AddressKeyInvalid extends Data.TaggedError("AddressKeyInvalid")<{
  readonly space: string
  readonly index: number
}> {}

export class AddressParentMismatch extends Data.TaggedError("AddressParentMismatch")<{
  readonly space: string
  readonly expectedParent: string
}> {}

export class AddressKeyNotSerializable extends Data.TaggedError("AddressKeyNotSerializable")<{
  readonly valueType: string
}> {}

export type AddressDecodeError =
  | AddressSegmentCountMismatch
  | AddressSpaceMismatch
  | AddressKeyInvalid

export type AddressConstructionError = AddressParentMismatch | AddressKeyNotSerializable

export interface AddressSpaceBase {
  readonly [AddressSpaceTypeId]: typeof AddressSpaceTypeId
  readonly name: string
  readonly [AddressSpaceParentTypeId]: AddressRoot | AddressSpaceBase
  readonly [AddressSpaceKeyTypeId]: unknown
  readonly [AddressSpaceEncodedKeyTypeId]: AddressEncodedValue
}

export interface AddressRef extends Equal.Equal {
  readonly [AddressTypeId]: typeof AddressTypeId
  readonly document: AddressDocument
}

type ParentOf<S extends AddressSpaceBase> = S[typeof AddressSpaceParentTypeId]

export type KeyOf<S extends AddressSpaceBase> = S[typeof AddressSpaceKeyTypeId]

export type EncodedKeyOf<S extends AddressSpaceBase> = S[typeof AddressSpaceEncodedKeyTypeId]

export type ParentAddress<Parent extends AddressRoot | AddressSpaceBase> =
  Parent extends AddressSpaceBase ? Address<Parent> : never

export type Address<S extends AddressSpaceBase> = AddressRef & {
  readonly space: S
  readonly key: KeyOf<S>
} & (ParentOf<S> extends AddressRoot
  ? { readonly parent?: never }
  : { readonly parent: ParentAddress<ParentOf<S>> })

export interface AddressSpace<
  Parent extends AddressRoot | AddressSpaceBase,
  Key,
  EncodedKey extends AddressEncodedValue,
  Name extends string = string,
> extends AddressSpaceBase {
  readonly name: Name
  readonly [AddressSpaceParentTypeId]: Parent
  readonly [AddressSpaceKeyTypeId]: Key
  readonly [AddressSpaceEncodedKeyTypeId]: EncodedKey
  readonly parentSpace: Parent
  readonly keySchema: Schema.Schema<Key, EncodedKey, never>
  readonly at: Parent extends AddressRoot
    ? (key: Key) => Address<this>
    : (parent: ParentAddress<Parent>, key: Key) => Address<this>
  readonly decode: (document: AddressDocument) => Effect.Effect<Address<this>, AddressDecodeError>
  readonly match: (reference: AddressRef) => Option.Option<Address<this>>
}

type RootSpaceOptions<Name extends string, Key, EncodedKey extends AddressEncodedValue> = {
  readonly name: Name
  readonly key: Schema.Schema<Key, EncodedKey, never>
}

type ChildSpaceOptions<Name extends string, Parent extends AddressSpaceBase, Key, EncodedKey extends AddressEncodedValue> = RootSpaceOptions<Name, Key, EncodedKey> & {
  readonly parent: Parent
}

const canonicalize = (value: unknown): string => {
  if (value === null) return "null"
  if (typeof value === "string") return JSON.stringify(value)
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (Array.isArray(value)) return `[${value.map(canonicalize).join(",")}]`
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalize(entry)}`)
    return `{${entries.join(",")}}`
  }
  throw new AddressKeyNotSerializable({ valueType: typeof value })
}

const documentToken = (document: AddressDocument): string =>
  document.segments
    .map(({ space, key }) => `${JSON.stringify(space)}:${canonicalize(key)}`)
    .join("/")

class AddressValue implements AddressRef {
  readonly [AddressTypeId]: typeof AddressTypeId = AddressTypeId

  constructor(
    readonly space: AddressSpaceBase,
    readonly key: unknown,
    readonly encodedKey: AddressEncodedValue,
    readonly parent: AddressRef | undefined,
    readonly document: AddressDocument,
    private readonly token: string,
  ) {}

  [Equal.symbol](other: Equal.Equal): boolean {
    return other instanceof AddressValue && this.space === other.space && this.token === other.token
  }

  [Hash.symbol](): number {
    return Hash.cached(this, Hash.string(this.token))
  }
}

const isAddressValue = (value: AddressRef): value is AddressValue => value instanceof AddressValue

const lineage = (space: AddressSpaceBase): readonly AddressSpaceBase[] => {
  const result: AddressSpaceBase[] = []
  let current: AddressRoot | AddressSpaceBase = space
  while (current !== AddressRoot) {
    const concrete = current as AddressSpaceBase
    result.unshift(concrete)
    current = (concrete as AddressSpace<AddressRoot | AddressSpaceBase, unknown, AddressEncodedValue>).parentSpace
  }
  return result
}

function makeSpace<const Name extends string, Key, EncodedKey extends AddressEncodedValue>(
  options: RootSpaceOptions<Name, Key, EncodedKey>,
): AddressSpace<AddressRoot, Key, EncodedKey, Name>
function makeSpace<const Name extends string, Parent extends AddressSpaceBase, Key, EncodedKey extends AddressEncodedValue>(
  options: ChildSpaceOptions<Name, Parent, Key, EncodedKey>,
): AddressSpace<Parent, Key, EncodedKey, Name>
function makeSpace<const Name extends string, Parent extends AddressRoot | AddressSpaceBase, Key, EncodedKey extends AddressEncodedValue>(
  options: RootSpaceOptions<Name, Key, EncodedKey> | ChildSpaceOptions<Name, Exclude<Parent, AddressRoot>, Key, EncodedKey>,
): AddressSpace<Parent, Key, EncodedKey, Name> {
  const parentSpace: Parent = "parent" in options ? options.parent as Parent : AddressRoot as Parent

  const build = (parent: AddressRef | undefined, key: Key, encodedKey: EncodedKey): AddressValue => {
    const segments = [
      ...(parent?.document.segments ?? []),
      { space: options.name, key: encodedKey },
    ]
    const document = { segments }
    return new AddressValue(space, key, encodedKey, parent, document, documentToken(document))
  }

  const construct = (...args: readonly unknown[]): AddressValue => {
    const parent = parentSpace === AddressRoot ? undefined : args[0] as AddressRef
    if (parentSpace !== AddressRoot) {
      const expectedParent = parentSpace as AddressSpaceBase
      if (parent === undefined || !isAddressValue(parent) || parent.space !== expectedParent) {
        throw new AddressParentMismatch({ space: options.name, expectedParent: expectedParent.name })
      }
    }
    const key = (parentSpace === AddressRoot ? args[0] : args[1]) as Key
    const encodedKey = Schema.encodeSync(options.key)(key)
    return build(parent, key, encodedKey)
  }

  const decode = (document: AddressDocument): Effect.Effect<AddressValue, AddressDecodeError> =>
    Effect.gen(function* () {
      const spaces = lineage(space)
      if (document.segments.length !== spaces.length) {
        return yield* new AddressSegmentCountMismatch({
          space: options.name,
          expected: spaces.length,
          actual: document.segments.length,
        })
      }

      let parent: AddressValue | undefined
      for (let index = 0; index < spaces.length; index += 1) {
        const expectedSpace = spaces[index] as AddressSpace<AddressRoot | AddressSpaceBase, unknown, AddressEncodedValue>
        const segment = document.segments[index]
        if (segment === undefined || segment.space !== expectedSpace.name) {
          return yield* new AddressSpaceMismatch({
            index,
            expected: expectedSpace.name,
            actual: segment?.space ?? null,
          })
        }
        const key = yield* Schema.decodeUnknown(expectedSpace.keySchema)(segment.key).pipe(
          Effect.mapError(() => new AddressKeyInvalid({ space: expectedSpace.name, index })),
        )
        parent = new AddressValue(
          expectedSpace,
          key,
          segment.key,
          parent,
          { segments: document.segments.slice(0, index + 1) },
          documentToken({ segments: document.segments.slice(0, index + 1) }),
        )
      }
      if (parent === undefined) {
        return yield* new AddressSegmentCountMismatch({ space: options.name, expected: 1, actual: 0 })
      }
      return parent
    })

  const space = {
    [AddressSpaceTypeId]: AddressSpaceTypeId,
    name: options.name,
    parentSpace,
    keySchema: options.key,
    at: construct,
    decode,
    match: (reference: AddressRef) =>
      isAddressValue(reference) && reference.space === space
        ? Option.some(reference)
        : Option.none(),
  } as unknown as AddressSpace<Parent, Key, EncodedKey, Name>

  return space
}

export const Address = {
  space: makeSpace,
  equals: (left: AddressRef, right: AddressRef): boolean => Equal.equals(left, right),
  isDescendantOf: (candidate: AddressRef, ancestor: AddressRef): boolean => {
    if (!isAddressValue(candidate)) return false
    let current: AddressRef | undefined = candidate
    while (current !== undefined) {
      if (Equal.equals(current, ancestor)) return true
      current = isAddressValue(current) ? current.parent : undefined
    }
    return false
  },
}

export const AddressInternal = {
  belongsTo: (reference: AddressRef, space: AddressSpaceBase): boolean =>
    isAddressValue(reference) && reference.space === space,
}

export type AddressTarget<A extends AddressRef> =
  | { readonly _tag: "None" }
  | { readonly _tag: "One"; readonly address: A }
  | { readonly _tag: "Many"; readonly addresses: readonly A[] }
  | { readonly _tag: "All" }

export const AddressTarget = {
  none: { _tag: "None" } as const,
  one: <A extends AddressRef>(address: A): AddressTarget<A> => ({ _tag: "One", address }),
  many: <A extends AddressRef>(addresses: readonly A[]): AddressTarget<A> => ({ _tag: "Many", addresses }),
  all: { _tag: "All" } as const,
}

export type AddressMatch<A extends AddressRef> =
  | { readonly _tag: "Exact"; readonly address: A }
  | { readonly _tag: "Subtree"; readonly address: A }

export const AddressMatch = {
  exact: <A extends AddressRef>(address: A): AddressMatch<A> => ({ _tag: "Exact", address }),
  subtree: <A extends AddressRef>(address: A): AddressMatch<A> => ({ _tag: "Subtree", address }),
  includes: (match: AddressMatch<AddressRef>, candidate: AddressRef): boolean =>
    match._tag === "Exact"
      ? Equal.equals(match.address, candidate)
      : Address.isDescendantOf(candidate, match.address),
}
