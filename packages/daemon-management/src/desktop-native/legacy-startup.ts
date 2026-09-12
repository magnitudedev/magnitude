import { Context, Effect, Layer, type Option, Schema } from "effect"
import { LegacyStartupFailed } from "./legacy-startup-command"
import { LegacyMacStartup, makeMacLegacyStartup } from "./legacy-startup-macos"
import { LegacyLinuxStartup, makeLinuxLegacyStartup } from "./legacy-startup-linux"

export const LegacyStartupRegistration = Schema.Union(LegacyMacStartup, LegacyLinuxStartup)
export type LegacyStartupRegistration = typeof LegacyStartupRegistration.Type
export interface LegacyStartup {
  readonly inspect: Effect.Effect<Option.Option<LegacyStartupRegistration>, LegacyStartupFailed>
  readonly unregister: (registration: LegacyStartupRegistration) => Effect.Effect<void, LegacyStartupFailed>
}
export const LegacyStartup = Context.GenericTag<LegacyStartup>("@magnitudedev/daemon-management/LegacyStartup")

export const macLegacyStartupLayer = (home: string, label?: string) => Layer.effect(LegacyStartup,
  makeMacLegacyStartup(home, label).pipe(Effect.map(adapter => LegacyStartup.of({
    inspect: adapter.inspect,
    unregister: registration => registration._tag === "MacLaunchAgent" ? adapter.unregister(registration)
      : Effect.fail(new LegacyStartupFailed({ message: "Saved legacy registration belongs to a different operating system" })),
  }))),
)
export const linuxLegacyStartupLayer = (options: Parameters<typeof makeLinuxLegacyStartup>[0]) => Layer.effect(LegacyStartup,
  makeLinuxLegacyStartup(options).pipe(Effect.map(adapter => LegacyStartup.of({
    inspect: adapter.inspect,
    unregister: registration => registration._tag === "SystemdUserService" ? adapter.unregister(registration)
      : Effect.fail(new LegacyStartupFailed({ message: "Saved legacy registration belongs to a different operating system" })),
  }))),
)
