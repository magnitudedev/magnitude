import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Effect, Option, Schema } from "effect"
import { relative } from "node:path"
import { appleRequirement } from "../../../release/src/trust"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { nativeInventory } from "../native-inventory"
import { command } from "../process"
import { admittedRuntimeComposition } from "../runtime-composition"

export const AppleTeamId = Schema.String.pipe(Schema.pattern(/^[A-Z0-9]{10}$/), Schema.brand("AppleTeamId"))
export const AppleTrustPolicy = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("development") }),
  Schema.Struct({ kind: Schema.Literal("production"), team: AppleTeamId }),
)
export const AppleCodeSignature = Schema.Struct({ path: Schema.String, identifier: Schema.NonEmptyString,
  kind: Schema.Literal("adhoc", "certificate"), team: Schema.optionalWith(Schema.String, { as: "Option", exact: true }) })
export const ApplePackageTrust = Schema.Struct({ policy: AppleTrustPolicy, productionTrusted: Schema.Boolean,
  signatures: Schema.Array(AppleCodeSignature), runtimeArtifacts: Schema.Array(Schema.String) })
const invalid = (message: string) => new AssertionFailure({ message: `Apple package trust: ${message}` })

export const appleTrustPolicy = (production: boolean, team: string | undefined) => !production
  ? Effect.succeed<typeof AppleTrustPolicy.Type>({ kind: "development" })
  : Schema.decodeUnknown(AppleTrustPolicy)({ kind: "production", team }).pipe(Effect.mapError(() =>
    new InfrastructureFailure({ operation: "package-trust", message: "Release verification requires a configured LAB_EXPECTED_APPLE_TEAM_ID" })))

/** Read-only verification: never re-sign a candidate or change the host trust store. */
export const verifyAppleSignature = (path: string, policy: typeof AppleTrustPolicy.Type, identifier?: string) => Effect.gen(function* () {
  const requirement = identifier ? appleRequirement(identifier, policy.kind === "production" ? policy.team : "")
    : policy.kind === "production" ? `anchor apple generic and certificate leaf[subject.OU] = "${policy.team}" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists` : undefined
  const verified = yield* command("/usr/bin/codesign", ["--verify", "--all-architectures", "--deep", "--strict",
    ...(requirement ? ["-R", `=${requirement}`] : []), path])
  if (verified.exitCode !== 0) return yield* invalid(`Invalid signature or publisher: ${path}: ${verified.stderr.slice(-2000)}`)
  const details = yield* command("/usr/bin/codesign", ["--display", "--verbose=4", path])
  if (details.exitCode !== 0) return yield* invalid(`Could not read signature identity: ${path}`)
  const output = details.stderr + "\n" + details.stdout
  const field = (name: string) => output.split("\n").filter(line => line.startsWith(name + "=")).map(line => line.slice(name.length + 1))
  const identifiers = field("Identifier"), teams = field("TeamIdentifier"), adhoc = field("Signature").includes("adhoc")
  if (identifiers.length !== 1 || !identifiers[0] || teams.length > 1 || (identifier && identifiers[0] !== identifier)) {
    return yield* invalid(`Missing or ambiguous signature identity: ${path}`)
  }
  const team = teams[0] && teams[0] !== "not set" ? Option.some(teams[0]) : Option.none<string>()
  if (!adhoc && field("Authority").length === 0) return yield* invalid(`Missing certificate authority: ${path}`)
  const timestamps = field("Timestamp")
  if (policy.kind === "production" && (adhoc || !Option.contains(team, policy.team) || timestamps.length !== 1 || !timestamps[0]?.trim())) {
    return yield* invalid(`Production code requires the expected team and a secure timestamp: ${path}`)
  }
  return AppleCodeSignature.make({ path, identifier: identifiers[0], kind: adhoc ? "adhoc" : "certificate", team })
})

export const verifyAppleNotarization = (app: string) => Effect.gen(function* () {
  for (const [executable, args] of [
    ["/usr/bin/xcrun", ["stapler", "validate", app]],
    ["/usr/sbin/spctl", ["--assess", "--type", "execute", "--verbose=4", app]],
  ] as const) {
    const result = yield* command(executable, args)
    if (result.exitCode !== 0) return yield* invalid(`Native notarization or Gatekeeper assessment rejected the installed app: ${result.stderr.slice(-2000)}`)
  }
})

export const inspectApplePackageTrust = (app: InstalledApplication, release: typeof ReleaseManifestSchema.Type,
  policy: typeof AppleTrustPolicy.Type) => Effect.scoped(Effect.gen(function* () {
  if (app.candidate.target.os !== "macos") return yield* invalid("Requires a macOS package")
  const signatures = [yield* verifyAppleSignature(app.root, policy, "dev.magnitude.desktop").pipe(
    Effect.map(value => ({ ...value, path: "application" })))]
  const runtime = yield* admittedRuntimeComposition(release, app.candidate.target)
  for (const [label, root] of [["application", app.root], ["runtime", runtime.root]] as const) {
    const files = yield* nativeInventory(root)
    if (!files.length) return yield* invalid(`No native code in ${label}`)
    for (const file of files) {
      if (file.image.format !== "mach-o") return yield* invalid(`Unexpected native format: ${file.path}`)
      signatures.push({ ...(yield* verifyAppleSignature(file.path, policy)), path: `${label}/${relative(root, file.path)}` })
    }
  }
  if (policy.kind === "production") yield* verifyAppleNotarization(app.root)
  return ApplePackageTrust.make({ policy, productionTrusted: policy.kind === "production", signatures, runtimeArtifacts: runtime.artifacts })
}))
