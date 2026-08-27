import * as CoreOperation from "./Operation.js"
import * as Mutation from "./Mutation.js"
import * as Query from "./Query.js"
import * as Subscription from "./Subscription.js"

/** @internal Runtime identity of a recursively composed operation group. */
export const TypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group")
/** @internal Exact members retained for adapter type derivation. */
export const MembersTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group/Members")
/** @internal Flattened operations in declaration order. */
export const OperationListTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group/OperationList")
/** @internal Every recursively contained definition in declaration order. */
export const DefinitionListTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group/DefinitionList")

export type AnyDefinition =
  | Query.Any
  | Mutation.Any
  | Subscription.Any

type Member = AnyDefinition | Any
type DefinitionOfMember<Value> = Value extends AnyDefinition
  ? Value
  : Value extends Group<Record<string, unknown>, infer ContainedDefinition, CoreOperation.Any>
    ? ContainedDefinition
    : never
type DefinitionOfMembers<Members extends Record<string, unknown>> = {
  readonly [Key in keyof Members]: DefinitionOfMember<Members[Key]>
}[keyof Members]

type ValidMembers<Members extends Record<string, unknown>> = {
  readonly [Key in keyof Members]: Members[Key] extends Member ? Members[Key] : never
}

export type Group<
  Members extends Record<string, unknown>,
  ContainedDefinition extends AnyDefinition,
  ContainedDeclaration extends CoreOperation.Any,
> = Members & {
  readonly [TypeId]: true
  readonly [MembersTypeId]?: Members
  readonly [DefinitionListTypeId]: ReadonlyArray<ContainedDefinition>
  readonly [OperationListTypeId]: ReadonlyArray<ContainedDeclaration>
}

export type Any = Group<Record<string, unknown>, AnyDefinition, CoreOperation.Any>
export type Definition<Value> = Value extends Group<Record<string, unknown>, infer ContainedDefinition, CoreOperation.Any>
  ? ContainedDefinition
  : never
export type Declaration<Value> = Value extends Group<Record<string, unknown>, AnyDefinition, infer ContainedDeclaration>
  ? ContainedDeclaration
  : never
/** Compatibility alias for the declared subset consumed by transport adapters. */
export type Operation<Value> = Declaration<Value>
/** The exact member record of a group: operations and nested groups by name. */
export type Members<Value> = Value extends { readonly [MembersTypeId]?: infer MemberRecord } ? MemberRecord : never

type ExtensionMember<Base, Added> = Base extends Any
  ? Added extends Any
    ? Group<
        ExtendedMembers<Members<Base>, Members<Added>>,
        Definition<Base> | Definition<Added>,
        Declaration<Base> | Declaration<Added>
      >
    : never
  : never

type ExtendedMembers<
  Base extends Record<string, unknown>,
  Added extends Record<string, unknown>,
> = {
  readonly [Key in keyof Base | keyof Added]: Key extends keyof Added
    ? Key extends keyof Base
      ? ExtensionMember<Base[Key], Added[Key]>
      : Added[Key]
    : Key extends keyof Base ? Base[Key] : never
}

type ValidExtension<Base extends Any, Added extends Record<string, unknown>> = {
  readonly [Key in keyof Added]: Key extends keyof Members<Base>
    ? Members<Base>[Key] extends Any
      ? Added[Key] extends Any ? Added[Key] : never
      : never
    : Added[Key] extends Member ? Added[Key] : never
}

export const isGroup = (value: unknown): value is Any =>
  typeof value === "object" && value !== null && TypeId in value

export const make = <const Members extends Record<string, unknown>>(
  members: Members & ValidMembers<Members>,
): Group<
  Members,
  Extract<DefinitionOfMembers<Members>, AnyDefinition>,
  Extract<DefinitionOfMembers<Members>, CoreOperation.Any>
> => {
  const definitions: AnyDefinition[] = []
  const operations: CoreOperation.Any[] = []
  const names = new Set<string>()

  const add = (definition: AnyDefinition) => {
    if (names.has(definition.name)) {
      throw new TypeError(`Duplicate operation ${definition.name}`)
    }
    names.add(definition.name)
    definitions.push(definition)
    if (CoreOperation.isDeclared(definition)) operations.push(definition)
  }

  for (const member of Object.values(members)) {
    if (Query.isQuery(member) || Mutation.isMutation(member) || Subscription.isSubscription(member)) {
      add(member)
      continue
    }
    if (isGroup(member)) {
      for (const definition of member[DefinitionListTypeId]) add(definition)
      continue
    }
    throw new TypeError("Group members must be Query, Mutation, Subscription, or Group values")
  }

  return Object.assign({}, members, {
    [TypeId]: true,
    [DefinitionListTypeId]: definitions,
    [OperationListTypeId]: operations,
  }) as never
}

/** Recursively extend a group without replacing existing operations or changing member shapes. */
export const extend = <
  Base extends Any,
  const Added extends Record<string, unknown>,
>(
  base: Base,
  added: Added & ValidMembers<Added> & ValidExtension<Base, Added>,
): Group<
  ExtendedMembers<Members<Base>, Added>,
  Definition<Base> | Extract<DefinitionOfMembers<Added>, AnyDefinition>,
  Declaration<Base> | Extract<DefinitionOfMembers<Added>, CoreOperation.Any>
> => {
  const members: Record<string, unknown> = Object.fromEntries(Object.entries(base))
  for (const [name, addition] of Object.entries(added)) {
    if (!Object.prototype.hasOwnProperty.call(base, name)) {
      members[name] = addition
      continue
    }
    const existing = base[name]
    if (isGroup(existing) && isGroup(addition)) {
      members[name] = extend(existing, addition as never)
      continue
    }
    if (isGroup(existing) || isGroup(addition)) {
      throw new TypeError(`Cannot extend Group member ${name} across a Group/operation shape conflict`)
    }
    throw new TypeError(`Cannot extend Group with duplicate operation member ${name}`)
  }
  return make(members as never) as never
}

export const definitions = <Value extends Any>(group: Value): ReadonlyArray<Definition<Value>> =>
  group[DefinitionListTypeId] as ReadonlyArray<Definition<Value>>

export const declarations = <Value extends Any>(group: Value): ReadonlyArray<Declaration<Value>> =>
  group[OperationListTypeId] as ReadonlyArray<Declaration<Value>>

export const operations = <Value extends Any>(group: Value): ReadonlyArray<Operation<Value>> =>
  declarations(group)

export const operation = <Value extends Any>(group: Value, name: string): Operation<Value> | undefined =>
  operations(group).find((candidate) => CoreOperation.declaration(candidate).name === name)
