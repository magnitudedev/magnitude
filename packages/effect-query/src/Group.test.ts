import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import { Group, Operation, Query } from "./index.js"

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

  it("rejects duplicate member and operation names", () => {
    const BaseQuery = Query.make("SameOperation", { payload: Empty, success: Schema.String })
    const base = Group.make({ BaseQuery })

    expect(() => Group.extend(base, { BaseQuery })).toThrow("duplicate member BaseQuery")
    expect(() => Group.extend(base, {
      Added: Group.make({
        OtherKey: Query.make("SameOperation", { payload: Empty, success: Schema.String }),
      }),
    })).toThrow("Duplicate operation SameOperation")
  })
})
