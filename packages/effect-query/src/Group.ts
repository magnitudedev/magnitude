import * as CoreOperation from "./Operation.js"

/** @internal Runtime identity of a recursively composed operation group. */
export const TypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group")
/** @internal Exact members retained for adapter type derivation. */
export const MembersTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group/Members")
/** @internal Flattened operations in declaration order. */
export const OperationListTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Group/OperationList")

type Member = CoreOperation.Any | Any
type OperationOfMember<Value> = Value extends CoreOperation.Any
  ? Value
  : Value extends Group<Record<string, unknown>, infer ContainedOperation>
    ? ContainedOperation
    : never
type OperationOfMembers<Members extends Record<string, unknown>> = {
  readonly [Key in keyof Members]: OperationOfMember<Members[Key]>
}[keyof Members]

type ValidMembers<Members extends Record<string, unknown>> = {
  readonly [Key in keyof Members]: Members[Key] extends Member ? Members[Key] : never
}

export type Group<Members extends Record<string, unknown>, ContainedOperation extends CoreOperation.Any> = Members & {
  readonly [TypeId]: true
  readonly [MembersTypeId]?: Members
  readonly [OperationListTypeId]: ReadonlyArray<ContainedOperation>
}

export type Any = Group<Record<string, unknown>, CoreOperation.Any>
export type Operation<Value> = Value extends Group<Record<string, unknown>, infer ContainedOperation>
  ? ContainedOperation
  : never

export const isGroup = (value: unknown): value is Any =>
  typeof value === "object" && value !== null && TypeId in value

export const make = <const Members extends Record<string, unknown>>(
  members: Members & ValidMembers<Members>,
): Group<Members, Extract<OperationOfMembers<Members>, CoreOperation.Any>> => {
  const operations: CoreOperation.Any[] = []
  const names = new Set<string>()

  const add = (operation: CoreOperation.Any) => {
    const declaration = CoreOperation.declaration(operation)
    if (names.has(declaration.name)) {
      throw new TypeError(`Duplicate operation ${declaration.name}`)
    }
    names.add(declaration.name)
    operations.push(operation)
  }

  for (const member of Object.values(members)) {
    if (CoreOperation.isDeclared(member)) {
      add(member)
      continue
    }
    if (isGroup(member)) {
      for (const operation of member[OperationListTypeId]) add(operation)
      continue
    }
    throw new TypeError("Group members must be declared Query, Mutation, Subscription, or Group values")
  }

  return Object.assign({}, members, {
    [TypeId]: true,
    [OperationListTypeId]: operations,
  }) as never
}

export const operations = <Value extends Any>(group: Value): ReadonlyArray<Operation<Value>> =>
  group[OperationListTypeId] as ReadonlyArray<Operation<Value>>

export const operation = <Value extends Any>(group: Value, name: string): Operation<Value> | undefined =>
  operations(group).find((candidate) => CoreOperation.declaration(candidate).name === name)
