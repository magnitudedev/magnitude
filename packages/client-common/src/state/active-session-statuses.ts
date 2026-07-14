import { useMemo } from "react"
import { Atom, useAtomMount, useAtomSet } from "@effect-atom/atom-react"
import { Cause, Effect, Fiber, Stream } from "effect"
import { RpcClient } from "@effect/rpc"
import {
  MagnitudeRpcs,
  type ActiveSessionStatus,
  type ActiveSessionStatuses,
  type ActiveSessionStatusesWireEvent,
} from "@magnitudedev/sdk"
import type { Layer } from "effect"
import { usePlatform } from "../platform/platform-context"

export type ActiveSessionStatusById = Readonly<Record<string, ActiveSessionStatus>>

export const activeSessionStatusesAtom = Atom.make<ActiveSessionStatusById>({})

interface ActiveSessionStatusCallbacks {
  readonly onSnapshot: (snapshot: ActiveSessionStatuses) => void
}

let currentFiber: Fiber.RuntimeFiber<void, unknown> | null = null

const isSnapshot = (event: ActiveSessionStatusesWireEvent): event is ActiveSessionStatuses =>
  !("_tag" in event && event._tag === "heartbeat")

const toStatusById = (snapshot: ActiveSessionStatuses): ActiveSessionStatusById => {
  const byId: Record<string, ActiveSessionStatus> = {}
  for (const status of snapshot.sessions) {
    byId[status.sessionId] = status
  }
  return byId
}

/**
 * Subscribe to the active session statuses stream using a shared protocol
 * layer. The fiber is managed externally — interrupt via
 * `interruptActiveSessionStatuses`.
 */
export function subscribeActiveSessionStatuses(
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
  callbacks: ActiveSessionStatusCallbacks,
): void {
  interruptActiveSessionStatuses()

  const effect = Effect.gen(function* () {
    const client = yield* RpcClient.make(MagnitudeRpcs)
    yield* client.StreamActiveSessionStatuses({}).pipe(
      Stream.filter(isSnapshot),
      Stream.tap((snapshot) => Effect.sync(() => callbacks.onSnapshot(snapshot))),
      Stream.runDrain,
    )
  }).pipe(
    Effect.catchAllCause((cause) =>
      Effect.logError(`StreamActiveSessionStatuses error: ${Cause.pretty(cause)}`),
    ),
    Effect.scoped,
    Effect.provide(protocolLayer),
  )

  currentFiber = Effect.runFork(effect)
}

export function interruptActiveSessionStatuses(): void {
  if (currentFiber) {
    Effect.runFork(Fiber.interrupt(currentFiber))
    currentFiber = null
  }
}

export function useActiveSessionStatusesSubscription(): void {
  const platform = usePlatform()
  const setStatuses = useAtomSet(activeSessionStatusesAtom)

  const subscriptionAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          subscribeActiveSessionStatuses(platform.protocolLayer, {
            onSnapshot: (snapshot) => setStatuses(toStatusById(snapshot)),
          })
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => {
              interruptActiveSessionStatuses()
              setStatuses({})
            }),
          )
        }),
      ),
    [platform.protocolLayer, setStatuses],
  )

  useAtomMount(subscriptionAtom)
}
