import { Context, Effect, Layer, Option, Schema, Scope } from "effect"
import { createRequire } from "node:module"

export class MacUpdateAdmissionFailed extends Schema.TaggedError<MacUpdateAdmissionFailed>()("MacUpdateAdmissionFailed", {}) {
  override get message() { return "The application installation could not be locked safely." }
}
export interface MacInstallationLease {
  readonly validate: Effect.Effect<void, MacUpdateAdmissionFailed>
}
export interface MacUpdateAdmission {
  readonly shared: (bundle: string) => Effect.Effect<Option.Option<MacInstallationLease>, MacUpdateAdmissionFailed, Scope.Scope>
  readonly exclusive: (bundle: string) => Effect.Effect<Option.Option<MacInstallationLease>, MacUpdateAdmissionFailed, Scope.Scope>
}
export const MacUpdateAdmission = Context.GenericTag<MacUpdateAdmission>("@magnitudedev/daemon-management/MacUpdateAdmission")
const attempt = <A>(run: () => A) => Effect.try({ try: run, catch: () => new MacUpdateAdmissionFailed() })

/** Scope retains installation admission independently of per-user application ownership. */
export const nativeMacUpdateAdmission = (addonPath: string) => Layer.effect(MacUpdateAdmission, Effect.gen(function* () {
  const native = yield* attempt(() => createRequire(import.meta.url)(addonPath) as {
    readonly acquireMacUpdateLease: (bundle: string, exclusive: boolean) => object | null
    readonly releaseMacUpdateLease: (lease: object) => void
    readonly validateMacUpdateLease: (lease: object) => void
  })
  const acquire = (bundle: string, exclusive: boolean) => Effect.acquireRelease(
    attempt(() => Option.fromNullable(native.acquireMacUpdateLease(bundle, exclusive))),
    lease => Effect.sync(() => { if (Option.isSome(lease)) native.releaseMacUpdateLease(lease.value) }))
  const validate = (lease: object) => attempt(() => native.validateMacUpdateLease(lease))
  return MacUpdateAdmission.of({
    shared: bundle => acquire(bundle, false).pipe(Effect.map(Option.map(lease => ({ validate: validate(lease) })))),
    exclusive: bundle => acquire(bundle, true).pipe(Effect.map(Option.map(lease => ({
      validate: validate(lease),
    })))),
  })
}))
