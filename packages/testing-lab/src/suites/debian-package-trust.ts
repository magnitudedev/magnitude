import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Effect, Schema } from "effect"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { checkedCommand } from "../process"
import { admittedRuntimeComposition } from "../runtime-composition"
import { PackagePayload, verifyDebPayload } from "./package-payload"

export const DebianPackageTrust = Schema.Struct({ policy: Schema.Literal("development"),
  signature: Schema.Literal("Unsigned"), productionTrusted: Schema.Literal(false),
  payload: PackagePayload, runtimeArtifacts: Schema.Array(Schema.String) })

/** Embedded signatures require their own publisher policy; absence is never publisher trust. */
export const inspectDebianSignature = (path: string) => Effect.gen(function* () {
  const result = yield* checkedCommand("ar", ["t", path], { timeoutMs: 30_000 })
  const members = result.stdout.trim().split(/\r?\n/).map(name => name.trim().replace(/\/$/, ""))
  if (members.some(name => name.startsWith("_gpg"))) return yield* new InfrastructureFailure({ operation: "package-trust",
    message: "This DEB contains an embedded signature; its expected publisher and verification policy are not configured" })
  if (members.length !== 3 || members[0] !== "debian-binary"
    || !/^control\.tar(?:\.(?:gz|xz|zst|bz2|lzma))?$/.test(members[1]!)
    || !/^data\.tar(?:\.(?:gz|xz|zst|bz2|lzma))?$/.test(members[2]!)) {
    return yield* new AssertionFailure({ message: "DEB archive has an unexpected member layout; refusing to classify it as an unsigned development package" })
  }
  return "Unsigned" as const
})

export const inspectDebianPackageTrust = (app: InstalledApplication, release: typeof ReleaseManifestSchema.Type, production: boolean) => Effect.scoped(Effect.gen(function* () {
  if (production) return yield* new InfrastructureFailure({ operation: "package-trust",
    message: "DEB production trust requires an independently configured publisher policy; unpublished package integrity cannot establish release trust" })
  const payload = yield* verifyDebPayload(app)
  const signature = yield* inspectDebianSignature(app.candidate.path)
  // Runtime composition downloads and rehashes every admitted base/backend archive before extraction.
  const runtime = yield* admittedRuntimeComposition(release, app.candidate.target)
  return DebianPackageTrust.make({ policy: "development", signature, productionTrusted: false,
    payload, runtimeArtifacts: runtime.artifacts })
}))
