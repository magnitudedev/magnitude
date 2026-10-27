import { describe, expect, it } from "vitest"
import { Effect, HashMap, Option, Schema } from "effect"
import { Address, AddressKeyInvalid, AddressMatch } from "../src"

describe("Address", () => {
  it("constructs value-equal schema-backed root addresses", () => {
    const Forks = Address.space({ name: "fork", key: Schema.NullOr(Schema.String) })
    const first = Forks.at("feature")
    const second = Forks.at("feature")

    expect(first).not.toBe(second)
    expect(Address.equals(first, second)).toBe(true)
    expect(HashMap.get(HashMap.make([first, 1]), second)).toEqual(Option.some(1))
    expect(Forks.at(null).key).toBeNull()
  })

  it("preserves typed ancestry and explicit subtree matching", () => {
    const Forks = Address.space({ name: "fork", key: Schema.String })
    const Turns = Address.space({ name: "turn", parent: Forks, key: Schema.Number })
    const fork = Forks.at("feature")
    const turn = Turns.at(fork, 2)

    expect(turn.parent.key).toBe("feature")
    expect(AddressMatch.includes(AddressMatch.exact(fork), turn)).toBe(false)
    expect(AddressMatch.includes(AddressMatch.subtree(fork), turn)).toBe(true)
  })

  it("round-trips through the concrete space and rejects another nominal space", async () => {
    const Models = Address.space({
      name: "model",
      key: Schema.Struct({ provider: Schema.String, name: Schema.String }),
    })
    const OtherModels = Address.space({
      name: "model",
      key: Schema.Struct({ provider: Schema.String, name: Schema.String }),
    })
    const original = Models.at({ provider: "local", name: "small" })
    const decoded = await Effect.runPromise(Models.decode(original.document))

    expect(decoded.key).toEqual(original.key)
    expect(Address.equals(decoded, original)).toBe(true)
    expect(Option.isNone(OtherModels.match(original))).toBe(true)
  })

  it("fails decoding through the expected space with a typed error", async () => {
    const Forks = Address.space({ name: "fork", key: Schema.String })
    const result = await Effect.runPromise(Effect.either(Forks.decode({
      segments: [{ space: "fork", key: 42 }],
    })))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left).toBeInstanceOf(AddressKeyInvalid)
  })
})
