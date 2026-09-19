import { Context, Effect, Option, Schema } from "effect"
import { join, resolve } from "node:path"
import { InfrastructureFailure } from "./domain"

/** Supplied only by guest composition after qualifying a dedicated OS user. Never inferred from HOME. */
export interface DisposableDesktopUser { readonly home: string }
export const DisposableDesktopUser = Context.GenericTag<DisposableDesktopUser>("@magnitudedev/testing-lab/DisposableDesktopUser")
export const DesktopEnvironment = Schema.Struct({ mode: Schema.Literal("isolated", "installed-user"), profile: Schema.String,
  port: Schema.Int.pipe(Schema.between(1024, 65535)), environment: Schema.Record({ key: Schema.String, value: Schema.String }) })

export const desktopEnvironment = (config: typeof DesktopEnvironment.Type) => Effect.gen(function* () {
  const fail = (message: string) => new InfrastructureFailure({ operation: "desktop-environment", message })
  const environment = { ...config.environment }
  if (config.mode === "isolated") {
    if (environment.MAGNITUDE_DEV_DATA_DIR !== config.profile || environment.MAGNITUDE_DEV_PORT !== String(config.port)) {
      return yield* fail("Isolated desktop profile and port must match the shared explicit application environment")
    }
  } else {
    const user = yield* Effect.serviceOption(DisposableDesktopUser)
    if (Option.isNone(user)) return yield* fail("Installed desktop mode requires a qualified disposable OS user")
    if (["MAGNITUDE_DEV_DATA_DIR", "MAGNITUDE_DEV_PORT", "MAGNITUDE_DESKTOP_STATE_DIR"].some(key => key in environment)) {
      return yield* fail("Installed desktop mode cannot contain development profile overrides")
    }
    if (config.port !== 10100 || resolve(config.profile) !== resolve(join(user.value.home, ".magnitude"))) {
      return yield* fail("Installed desktop mode must use the normal user data directory and port")
    }
    if (environment.HOME !== user.value.home || (process.platform === "win32" && environment.USERPROFILE !== user.value.home)) {
      return yield* fail("Installed desktop environment differs from the qualified OS user")
    }
  }
  environment.MAGNITUDE_SHELL_ENV_INHERITED = "1"
  return environment
})
