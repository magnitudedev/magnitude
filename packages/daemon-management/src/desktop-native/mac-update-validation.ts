import { APPLE_TEAM_ID, appleRequirement } from "@magnitudedev/release/trust"
import { Context, Effect, Layer, Schema } from "effect"
import { createRequire } from "node:module"

export const MacBundleExpectation = Schema.Struct({
  version: Schema.NonEmptyString,
  architecture: Schema.Literal("arm64", "x64"),
})
export type MacBundleExpectation = typeof MacBundleExpectation.Type

export class MacBundleVerificationFailed extends Schema.TaggedError<MacBundleVerificationFailed>()("MacBundleVerificationFailed", {}) {}
export interface MacBundleVerifier {
  readonly verify: (path: string, expected: MacBundleExpectation) => Effect.Effect<void, MacBundleVerificationFailed>
}
export const MacBundleVerifier = Context.GenericTag<MacBundleVerifier>("@magnitudedev/daemon-management/MacBundleVerifier")

/** The owner retains exclusive staging access until installation finishes. */
export const nativeMacBundleVerifier = (addonPath: string) => Layer.effect(MacBundleVerifier, Effect.gen(function* () {
  // Update verification cannot fall back to identifier-only trust in an unsigned build.
  if (!/^[A-Z0-9]{10}$/.test(APPLE_TEAM_ID)) return yield* new MacBundleVerificationFailed()
  const requirement = appleRequirement("dev.magnitude.desktop", APPLE_TEAM_ID)
  const bindings = yield* Effect.try({
    try: () => createRequire(import.meta.url)(addonPath) as {
      readonly verifyMacBundle: (path: string, requirement: string, version: string, architecture: string) => Promise<void>
    },
    catch: () => new MacBundleVerificationFailed(),
  })
  return MacBundleVerifier.of({
    verify: (path, expected) => Effect.tryPromise({
      try: () => bindings.verifyMacBundle(path, requirement, expected.version, expected.architecture === "x64" ? "x86_64" : "arm64"),
      catch: () => new MacBundleVerificationFailed(),
    // Wait for native validation to release its resources before staging cleanup on cancellation.
    }).pipe(Effect.uninterruptible),
  })
}))
