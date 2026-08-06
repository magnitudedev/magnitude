import type { ClientId } from "@magnitudedev/acn-protocol"

export interface ClientLease {
  readonly renewalGeneration: number
  readonly expiresAtNanos: bigint
}

export interface ClientLeaseSet {
  readonly nextRenewalGeneration: number
  readonly leases: ReadonlyMap<ClientId, ClientLease>
}

export const emptyClientLeaseSet = (): ClientLeaseSet => ({
  nextRenewalGeneration: 0,
  leases: new Map(),
})

export interface RenewClientLeaseTransition {
  readonly state: ClientLeaseSet
  readonly connectionChanged: boolean
  readonly renewalGeneration: number
}

export const renewClientLease = (
  current: ClientLeaseSet,
  clientId: ClientId,
  nowNanos: bigint,
  leaseTimeoutNanos: bigint
): RenewClientLeaseTransition => {
  const renewalGeneration = current.nextRenewalGeneration + 1
  const leases = new Map(current.leases)
  leases.set(clientId, {
    renewalGeneration,
    expiresAtNanos: nowNanos + leaseTimeoutNanos,
  })
  return {
    state: {
      nextRenewalGeneration: renewalGeneration,
      leases,
    },
    connectionChanged: current.leases.size === 0,
    renewalGeneration,
  }
}

export interface RemoveClientLeaseTransition {
  readonly state: ClientLeaseSet
  readonly removed: boolean
  readonly connectionChanged: boolean
}

export const removeClientLease = (
  current: ClientLeaseSet,
  clientId: ClientId,
  expectedRenewalGeneration?: number
): RemoveClientLeaseTransition => {
  const lease = current.leases.get(clientId)
  if (
    lease === undefined ||
    (expectedRenewalGeneration !== undefined && lease.renewalGeneration !== expectedRenewalGeneration)
  ) {
    return { state: current, removed: false, connectionChanged: false }
  }
  const leases = new Map(current.leases)
  leases.delete(clientId)
  return {
    state: {
      ...current,
      leases,
    },
    removed: true,
    connectionChanged: leases.size === 0,
  }
}

export const dueClientLeases = (
  current: ClientLeaseSet,
  nowNanos: bigint
): ReadonlyArray<readonly [ClientId, ClientLease]> =>
  [...current.leases].filter(([, lease]) => lease.expiresAtNanos <= nowNanos)

export const nextClientLeaseDeadline = (current: ClientLeaseSet): bigint | undefined => {
  let earliest: bigint | undefined
  for (const lease of current.leases.values()) {
    if (earliest === undefined || lease.expiresAtNanos < earliest) {
      earliest = lease.expiresAtNanos
    }
  }
  return earliest
}
