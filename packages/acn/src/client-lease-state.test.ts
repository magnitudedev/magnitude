import { ClientIdSchema } from "@magnitudedev/acn-protocol"
import { describe, expect, it } from "vitest"
import {
  dueClientLeases,
  emptyClientLeaseSet,
  nextClientLeaseDeadline,
  removeClientLease,
  renewClientLease,
} from "./client-lease-state"

const clientA = ClientIdSchema.make("client-a")
const clientB = ClientIdSchema.make("client-b")

describe("client lease state", () => {
  it("renews one exact client without increasing the count", () => {
    const first = renewClientLease(emptyClientLeaseSet(), clientA, 10n, 35n)
    const second = renewClientLease(first.state, clientA, 20n, 35n)

    expect(first.connectionChanged).toBe(true)
    expect(second.connectionChanged).toBe(false)
    expect(second.state.leases.size).toBe(1)
    expect(second.state.leases.get(clientA)).toEqual({
      renewalGeneration: 2,
      expiresAtNanos: 55n,
    })
  })

  it("rejects stale expiry generations", () => {
    const first = renewClientLease(emptyClientLeaseSet(), clientA, 0n, 35n)
    const second = renewClientLease(first.state, clientA, 15n, 35n)
    const stale = removeClientLease(second.state, clientA, first.renewalGeneration)

    expect(stale.removed).toBe(false)
    expect(stale.state).toBe(second.state)
  })

  it("identifies only due leases and the nearest deadline", () => {
    const first = renewClientLease(emptyClientLeaseSet(), clientA, 0n, 35n)
    const second = renewClientLease(first.state, clientB, 10n, 35n)

    expect(nextClientLeaseDeadline(second.state)).toBe(35n)
    expect(dueClientLeases(second.state, 34n)).toEqual([])
    expect(dueClientLeases(second.state, 35n).map(([id]) => id)).toEqual([clientA])
  })
})
