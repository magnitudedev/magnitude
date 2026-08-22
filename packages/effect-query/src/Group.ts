import * as Operation from "./Operation.js"

/** @internal Runtime identity of a recursively composed operation group. */
export const TypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group")
/** @internal Exact members retained for adapter type derivation. */
export const MembersTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group/Members")
/** @internal Flattened operations in declaration order. */
export const OperationsTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group/Operations")

type Member = Operation.Any | Any
type OperationsOfMember<Value> = Value extends Operation.Any
  ? Value
  : Value extends Group<Record<string, unknown>, infer Operations>
    ? Operations
    : never
type OperationsOfMembers<Members extends Record<string, unknown>> = {
  readonly [Key in keyof Members]: OperationsOfMember<Members[Key]>
}[keyof Members]

type ValidMembers<Members extends Record<string, unknown>> = {
  readonly [Key in keyof Members]: Members[Key] extends Member ? Members[Key] : never
}

export type Group<Members extends Record<string, unknown>, Operations extends Operation.Any> = Members & {
  readonly [TypeId]: true
  readonly [MembersTypeId]?: Members
  readonly [OperationsTypeId]: ReadonlyArray<Operations>
}

export type Any = Group<Record<string, unknown>, Operation.Any>
export type Operations<Value> = Value extends Group<Record<string, unknown>, infer ValueOperations>
  ? ValueOperations
  : never

export const isGroup = (value: unknown): value is Any =>
  typeof value === "object" && value !== null && TypeId in value

export const make = <const Members extends Record<string, unknown>>(
  members: Members & ValidMembers<Members>,
): Group<Members, Extract<OperationsOfMembers<Members>, Operation.Any>> => {
  const operations: Operation.Any[] = []
  const names = new Set<string>()

  const add = (operation: Operation.Any) => {
    const declaration = Operation.declaration(operation)
    if (names.has(declaration.name)) {
      throw new TypeError(`Duplicate operation ${declaration.name}`)
    }
    names.add(declaration.name)
    operations.push(operation)
  }

  for (const member of Object.values(members)) {
    if (Operation.isDeclared(member)) {
      add(member)
      continue
    }
    if (isGroup(member)) {
      for (const operation of member[OperationsTypeId]) add(operation)
      continue
    }
    throw new TypeError("Group members must be declared Query, Mutation, Subscription, or Group values")
  }

  return Object.assign({}, members, {
    [TypeId]: true,
    [OperationsTypeId]: operations,
  }) as never
}

export const operations = <Value extends Any>(group: Value): ReadonlyArray<Operations<Value>> =>
  group[OperationsTypeId] as ReadonlyArray<Operations<Value>>

export const operation = <Value extends Any>(group: Value, name: string): Operations<Value> | undefined =>
  operations(group).find((candidate) => Operation.declaration(candidate).name === name)
