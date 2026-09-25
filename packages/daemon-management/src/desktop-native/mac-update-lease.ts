import { Context, Effect, Layer, Option, Schema, Scope } from "effect"
import { createRequire } from "node:module"

export class MacUpdateAdmissionFailed extends Schema.TaggedError<MacUpdateAdmissionFailed>()("MacUpdateAdmissionFailed", {}) {
  override get message() { return "The application installation could not be locked safely." }
}
export interface MacInstallationLease {
  readonly validate: Effect.Effect<void, MacUpdateAdmissionFailed>
}
export const MAC_UPDATE_LEASE_DESCRIPTOR = "MAGNITUDE_MAC_UPDATE_LEASE_FD"
export interface MacExclusiveInstallationLease extends MacInstallationLease {
  readonly bundle: string
  readonly replaceProcess: (executable: string, args: readonly string[], environment: Readonly<Record<string, string | undefined>>) => Effect.Effect<never, MacUpdateAdmissionFailed>
}
export interface MacUpdateAdmission {
  readonly shared: (bundle: string) => Effect.Effect<Option.Option<MacInstallationLease>, MacUpdateAdmissionFailed, Scope.Scope>
  readonly exclusive: (bundle: string) => Effect.Effect<Option.Option<MacExclusiveInstallationLease>, MacUpdateAdmissionFailed, Scope.Scope>
  readonly adopt: (bundle: string, descriptor: number) => Effect.Effect<MacExclusiveInstallationLease, MacUpdateAdmissionFailed, Scope.Scope>
}
export const MacUpdateAdmission = Context.GenericTag<MacUpdateAdmission>("@magnitudedev/daemon-management/MacUpdateAdmission")
const attempt = <A>(run: () => A) => Effect.try({ try: run, catch: () => new MacUpdateAdmissionFailed() })

/** Scope retains installation admission independently of per-user application ownership. */
export const nativeMacUpdateAdmission = (addonPath: string) => Layer.effect(MacUpdateAdmission, Effect.gen(function* () {
  const native = yield* attempt(() => createRequire(import.meta.url)(addonPath) as {
    readonly acquireMacUpdateLease: (bundle: string, exclusive: boolean) => object | null
    readonly adoptMacUpdateLease: (bundle: string, descriptor: number) => object
    readonly prepareMacUpdateLeaseExec: (lease: object) => number
    readonly cancelMacUpdateLeaseExec: (lease: object) => void
    readonly replaceProcess: (executable: string, args: readonly string[], environment: readonly string[]) => never
    readonly releaseMacUpdateLease: (lease: object) => void
    readonly validateMacUpdateLease: (lease: object) => void
  })
  const acquire = (bundle: string, exclusive: boolean) => Effect.acquireRelease(
    attempt(() => Option.fromNullable(native.acquireMacUpdateLease(bundle, exclusive))),
    lease => Effect.sync(() => { if (Option.isSome(lease)) native.releaseMacUpdateLease(lease.value) }))
  const validate = (lease: object) => attempt(() => native.validateMacUpdateLease(lease))
  const exclusiveLease = (lease: object, bundle: string): MacExclusiveInstallationLease => ({
    bundle,
    validate: validate(lease),
    replaceProcess: (executable, args, environment) => attempt(() => {
      const descriptor = native.prepareMacUpdateLeaseExec(lease)
      try {
        const inherited = { ...environment, [MAC_UPDATE_LEASE_DESCRIPTOR]: String(descriptor) }
        return native.replaceProcess(executable, args, Object.entries(inherited).flatMap(([key, value]) => value === undefined ? [] : [`${key}=${value}`]))
      } finally { native.cancelMacUpdateLeaseExec(lease) }
    }),
  })
  return MacUpdateAdmission.of({
    shared: bundle => acquire(bundle, false).pipe(Effect.map(Option.map(lease => ({ validate: validate(lease) })))),
    exclusive: bundle => acquire(bundle, true).pipe(Effect.map(Option.map(lease => exclusiveLease(lease, bundle)))),
    adopt: (bundle, descriptor) => Effect.acquireRelease(attempt(() => native.adoptMacUpdateLease(bundle, descriptor)),
      lease => Effect.sync(() => native.releaseMacUpdateLease(lease))).pipe(Effect.map(lease => exclusiveLease(lease, bundle))),
  })
}))

/** Retained by each installed application owner until its service tree has retired. */
export const acquireMacApplicationInstallationLease = (bundle: string) => Effect.gen(function* () {
  const admission = yield* MacUpdateAdmission
  const lease = yield* admission.shared(bundle)
  if (Option.isNone(lease)) return yield* new MacUpdateAdmissionFailed()
  yield* lease.value.validate
})
