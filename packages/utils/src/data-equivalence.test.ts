import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { dataEquivalence } from "./data-equivalence"

class Location extends Schema.Class<Location>("Location")({ path: Schema.String }) {}
const Model = Schema.Union(
  Schema.TaggedStruct("Catalog", {
    id: Schema.String,
    license: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    location: Schema.optionalWith(Location, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Custom", { id: Schema.String, bytes: Schema.Number }),
)
const State = Schema.Struct({ models: Schema.Array(Model) })
const decode = Schema.decodeUnknownSync(State)
const encoded = {
  models: [
    { _tag: "Catalog", id: "a", license: "mit", location: { path: "/models/a" } },
    { _tag: "Custom", id: "b", bytes: 1 },
  ],
}

describe("dataEquivalence", () => {
  const equivalent = dataEquivalence<typeof State.Type>()

  it("treats two separate reads of the same data as equal, as Schema.equivalence does", () => {
    const left = decode(encoded)
    const right = decode(structuredClone(encoded))
    expect(equivalent(left, right)).toBe(true)
    expect(Schema.equivalence(State)(left, right)).toBe(true)
  })

  it("sees a change inside an Option, a class or a union member", () => {
    const base = decode(encoded)
    const changed = [
      { ...encoded, models: [{ ...encoded.models[0], license: "apache-2.0" }, encoded.models[1]] },
      { ...encoded, models: [{ ...encoded.models[0], location: { path: "/models/z" } }, encoded.models[1]] },
      { ...encoded, models: [{ _tag: "Catalog", id: "a", location: { path: "/models/a" } }, encoded.models[1]] },
      { ...encoded, models: [encoded.models[0], { _tag: "Custom", id: "b", bytes: 2 }] },
    ]
    for (const next of changed) expect(equivalent(base, decode(next))).toBe(false)
  })

  it("tells Some from None", () => {
    expect(dataEquivalence<Option.Option<number>>()(Option.some(1), Option.none())).toBe(false)
    expect(dataEquivalence<Option.Option<number>>()(Option.none(), Option.none())).toBe(true)
  })
})
