import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { userInfo } from "node:os"
import { isAbsolute, win32 } from "node:path"
import { InfrastructureFailure, Provider } from "./domain"
import type { DisposableDesktopUser } from "./desktop-environment"

export const GuestUserIdentity = Schema.Struct({ home: Schema.NonEmptyString, username: Schema.NonEmptyString, uid: Schema.Int })
export const admitGuestUser = (provider: typeof Provider.Type, disposable: boolean, identity: typeof GuestUserIdentity.Type,
  environment: Readonly<Record<string, string>>, platform: NodeJS.Platform = process.platform, isolatedContainer = false) => Effect.gen(function* () {
  if (!disposable) return Option.none<DisposableDesktopUser>()
  const fail = (message: string) => new InfrastructureFailure({ operation: "guest-user", message })
  if (provider !== "azure" && provider !== "namespace" && !(provider === "spark" && platform === "linux" && isolatedContainer)) return yield* fail("Native desktop user context requires a disposable cloud allocation or verified lab container")
  if (!(platform === "win32" ? win32.isAbsolute(identity.home) : isAbsolute(identity.home)) || (platform !== "win32" && identity.uid <= 0)) return yield* fail("Native desktop context requires a non-root OS user with an absolute home")
  if ((platform !== "win32" || environment.HOME !== undefined) && environment.HOME !== identity.home || (platform === "win32" && environment.USERPROFILE !== identity.home)) {
    return yield* fail("Guest home environment does not match the operating system account")
  }
  return Option.some<DisposableDesktopUser>({ home: identity.home })
})

/** The invocation's disposable claim comes from the coordinator's allocation, not from source inputs. */
export const qualifyGuestUser = (provider: typeof Provider.Type, disposable: boolean, environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  if (!disposable) return Option.none<DisposableDesktopUser>()
  const native = yield* Effect.try({ try: () => userInfo(), catch: () => new InfrastructureFailure({ operation: "guest-user", message: "Cannot inspect the actual OS account" }) })
  const identity = GuestUserIdentity.make({ home: native.homedir, username: native.username, uid: native.uid })
  const fs = yield* FileSystem.FileSystem
  const isolatedContainer = provider === "spark" && process.platform === "linux" && (yield* fs.exists("/.dockerenv"))
  const admitted = yield* admitGuestUser(provider, disposable, identity, environment, process.platform, isolatedContainer)
  if (Option.isSome(admitted)) {
    const home = yield* fs.stat(admitted.value.home)
    if (home.type !== "Directory" || (process.platform !== "win32" && Option.getOrUndefined(home.uid) !== native.uid)) {
      return yield* new InfrastructureFailure({ operation: "guest-user", message: "Guest home directory is not owned by the actual OS account" })
    }
  }
  return admitted
})
