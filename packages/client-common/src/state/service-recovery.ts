import { useMemo } from "react"
import {
  Atom,
  Registry,
  Result,
  useAtomMount,
  useAtomValue,
} from "@effect-atom/atom-react"
import { ServiceRecoveryInactive, type ServiceRecoveryState } from "../connection/connection"
import { Effect, Option, Stream } from "effect"
import { pushNotificationAtom } from "./notification-area-state"
import { useServiceStartup } from "./service-startup"

export function useServiceRecoveryState(): ServiceRecoveryState {
  const recovery = useServiceStartup().recovery
  const stateAtom = useMemo(
    () => Atom.make(recovery.changes, {
      initialValue: new ServiceRecoveryInactive({}),
    }),
    [recovery],
  )
  const completionAtom = useMemo(
    () => Atom.make(Effect.gen(function* () {
      const registry = yield* Registry.AtomRegistry
      yield* recovery.changes.pipe(Stream.runForEach((state) =>
        state._tag === "Recovered"
          ? Effect.sync(() => {
              registry.set(pushNotificationAtom, {
                message: "Reconnected to Magnitude service",
                priority: "notice",
                action: Option.none(),
                dismissAfterMilliseconds: 5_000,
              })
            })
          : Effect.void))
    })),
    [recovery],
  )
  useAtomMount(completionAtom)
  return Option.getOrElse(
    Result.value(useAtomValue(stateAtom)),
    () => new ServiceRecoveryInactive({}),
  )
}
