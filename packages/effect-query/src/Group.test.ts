import { Effect, Schema } from "effect"
import { describe, expect, expectTypeOf, it } from "vitest"
import { Client, Group, Mutation, Operation, Query } from "./index.js"

const Empty = Schema.Struct({})

describe("Group.extend", () => {
  it("preserves the base shape and flattens added groups exactly once", () => {
    const BaseQuery = Query.make("BaseQuery", { payload: Empty, success: Schema.String })
    const AddedQuery = Query.make("AddedQuery", { payload: Empty, success: Schema.Number })
    const base = Group.make({ BaseQuery })
    const added = Group.make({ AddedQuery })

    const combined = Group.extend(base, { Added: added })

    expect(combined.BaseQuery).toBe(BaseQuery)
    expect(combined.Added).toBe(added)
    expect(Group.operations(combined).map((operation) => Operation.declaration(operation).name))
      .toEqual(["BaseQuery", "AddedQuery"])
  })

  it("recursively extends existing domain groups", () => {
    const Remote = Query.make("Remote", { payload: Empty, success: Schema.String })
    const Composed = Mutation.make("Composed", {
      effect: () => Effect.succeed("done"),
    })
    const base = Group.make({ Models: Group.make({ Remote }) })

    const combined = Group.extend(base, {
      Models: Group.make({ Composed }),
    })

    expect(combined.Models.Remote).toBe(Remote)
    expect(combined.Models.Composed).toBe(Composed)
    expect(Group.definitions(combined).map(({ name }) => name)).toEqual(["Remote", "Composed"])
    expect(Group.declarations(combined).map((operation) => Operation.declaration(operation).name))
      .toEqual(["Remote"])
    expectTypeOf<Group.Definition<typeof combined>>().toEqualTypeOf<typeof Remote | typeof Composed>()
    expectTypeOf<Group.Declaration<typeof combined>>().toEqualTypeOf<typeof Remote>()
  })

  it("rejects duplicate member and operation names", () => {
    const BaseQuery = Query.make("SameOperation", { payload: Empty, success: Schema.String })
    const base = Group.make({ BaseQuery })

    expect(() => Group.extend(base, { BaseQuery } as never)).toThrow("duplicate operation member BaseQuery")
    expect(() => Group.extend(base, {
      Added: Group.make({
        OtherKey: Query.make("SameOperation", { payload: Empty, success: Schema.String }),
      }),
    })).toThrow("Duplicate operation SameOperation")
  })

  it("rejects Group/operation shape conflicts", () => {
    const BaseQuery = Query.make("BaseQuery", { payload: Empty, success: Schema.String })
    const base = Group.make({ Models: Group.make({ BaseQuery }) })
    const replacement = Query.make("Replacement", { payload: Empty, success: Schema.String })

    expect(() => Group.extend(base, { Models: replacement } as never))
      .toThrow("Group/operation shape conflict")
  })

  it("requires implementations for every declared member", () => {
    const First = Query.make("First", { payload: Empty, success: Schema.String })
    const Second = Query.make("Second", { payload: Empty, success: Schema.String })
    const group = Group.make({ First, Second })

    type Complete = Operation.Implementations<"First" | "Second", any>
    type Incomplete = Operation.Implementations<"First", any>

    expectTypeOf<Client.Materializable<typeof group, Complete>>().toEqualTypeOf<unknown>()
    expectTypeOf<Client.Materializable<typeof group, Incomplete>>().not.toEqualTypeOf<unknown>()
  })

  it("requires services used by Effect-backed members", () => {
    class RequiredService extends Effect.Tag("RequiredService")<RequiredService, { readonly value: true }>() {}
    const Local = Mutation.make("Local", {
      effect: () => Effect.asVoid(RequiredService),
    })
    const group = Group.make({ Local })

    expectTypeOf<Client.Materializable<typeof group, RequiredService>>().toEqualTypeOf<unknown>()
    expectTypeOf<Client.Materializable<typeof group, never>>().not.toEqualTypeOf<unknown>()
  })
})
