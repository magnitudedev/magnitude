import { describe, expect, it } from "vitest"
import { makeResidentWatchRegistry } from "./reactive-rpc-watch-registry"

describe("resident reactive RPC watches", () => {
  it("opens one watch per client and resource", () => {
    const registry = makeResidentWatchRegistry<string>()
    const client = {}
    let opens = 0
    const first = registry.getOrCreate(client, "catalog", () => ({ id: ++opens }))
    const second = registry.getOrCreate(client, "catalog", () => ({ id: ++opens }))
    const slots = registry.getOrCreate(client, "slots", () => ({ id: ++opens }))

    expect(second).toBe(first)
    expect(slots).not.toBe(first)
    expect(opens).toBe(2)
  })
})
