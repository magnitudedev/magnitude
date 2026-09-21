import { isWindows } from "../domain"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { join, relative } from "node:path"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { nativeInventory } from "../native-inventory"
import { command } from "../process"
import { admittedRuntimeComposition } from "../runtime-composition"

export const WindowsPublisher = Schema.NonEmptyString.pipe(Schema.brand("WindowsPublisher"))
export const WindowsTrustPolicy = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("development") }),
  Schema.Struct({ kind: Schema.Literal("production"), publisher: WindowsPublisher, signtool: Schema.NonEmptyString }),
)
export const WindowsSignatureObservation = Schema.Struct({
  status: Schema.Literal("Valid", "NotSigned", "HashMismatch", "NotTrusted", "UnknownError", "NotSupportedFileFormat", "Incompatible"),
  signatureType: Schema.Literal("None", "Authenticode", "Catalog"),
  publisher: Schema.optionalWith(WindowsPublisher, { as: "Option", exact: true }), timestamped: Schema.Boolean,
})
export const WindowsCodeSignature = Schema.Struct({ path: Schema.String, ...WindowsSignatureObservation.fields })
export const WindowsPackageTrust = Schema.Struct({ policy: WindowsTrustPolicy, productionTrusted: Schema.Boolean,
  signatures: Schema.Array(WindowsCodeSignature), runtimeArtifacts: Schema.Array(Schema.String) })
const invalid = (message: string) => new AssertionFailure({ message: `Windows package trust: ${message}` })

export const windowsTrustPolicy = (production: boolean, environment: Readonly<Record<string, string>>) => !production
  ? Effect.succeed<typeof WindowsTrustPolicy.Type>({ kind: "development" })
  : Schema.decodeUnknown(WindowsTrustPolicy)({ kind: "production", publisher: environment.LAB_EXPECTED_WINDOWS_PUBLISHER?.trim(),
    signtool: environment.LAB_WINDOWS_SIGNTOOL?.trim() }).pipe(Effect.mapError(() => new InfrastructureFailure({ operation: "package-trust",
      message: "Release verification requires LAB_EXPECTED_WINDOWS_PUBLISHER and LAB_WINDOWS_SIGNTOOL in the worker configuration" })))

export const windowsSignatureScript = String.raw`
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
if (!(Test-Path -LiteralPath $env:LAB_SIGNATURE_PATH -PathType Leaf)) { throw 'Signature input is not a file' }
$signature = Get-AuthenticodeSignature -LiteralPath $env:LAB_SIGNATURE_PATH
$record = @{ status = [string]$signature.Status; signatureType = [string]$signature.SignatureType; timestamped = [bool]$signature.TimeStamperCertificate }
if ($signature.SignerCertificate) {
  $record.publisher = $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false)
}
$record | ConvertTo-Json -Compress
`

/** Native trust inspection uses literal paths, never embedded executable source. */
export const verifyWindowsSignature = (path: string, policy: typeof WindowsTrustPolicy.Type,
  environment: Readonly<Record<string, string>>, microsoftRuntime = false) => Effect.gen(function* () {
  const result = yield* command("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", windowsSignatureScript], {
    env: { ...environment, LAB_SIGNATURE_PATH: path }, inheritEnv: false,
  })
  if (result.exitCode !== 0) return yield* new InfrastructureFailure({ operation: "package-trust", message: `Authenticode inspection failed: ${result.stderr.slice(-2000)}` })
  const observed = yield* Schema.decodeUnknown(Schema.parseJson(WindowsSignatureObservation))(result.stdout.trim()).pipe(
    Effect.mapError(() => new InfrastructureFailure({ operation: "package-trust", message: "Invalid Authenticode inspection response" })))
  if (observed.status !== "Valid" && observed.status !== "NotSigned") return yield* invalid(`${observed.status}: ${path}`)
  if (observed.status === "Valid" && (Option.isNone(observed.publisher) || observed.signatureType === "None")) return yield* invalid(`Valid signature has no publisher or signature type: ${path}`)
  if (observed.status === "NotSigned" && (Option.isSome(observed.publisher) || observed.timestamped || observed.signatureType !== "None")) return yield* invalid(`Inconsistent unsigned signature response: ${path}`)
  if (policy.kind === "production") {
    const publisher = microsoftRuntime ? WindowsPublisher.make("Microsoft Windows Software Compatibility Publisher") : policy.publisher
    if (observed.status !== "Valid" || !Option.contains(observed.publisher, publisher) || !observed.timestamped) {
      return yield* invalid(`Expected a timestamped signature from ${publisher}: ${path}`)
    }
    const verified = yield* command(policy.signtool, ["verify", "/pa", "/all", "/tw", path], { env: environment, inheritEnv: false })
    if (verified.exitCode !== 0) return yield* invalid(`SignTool rejected a signature or timestamp: ${path}: ${verified.stderr.slice(-2000)}`)
  }
  return WindowsCodeSignature.make({ path, ...observed })
})

export const inspectWindowsPackageTrust = (app: InstalledApplication, release: typeof ReleaseManifestSchema.Type,
  policy: typeof WindowsTrustPolicy.Type, environment: Readonly<Record<string, string>>) => Effect.scoped(Effect.gen(function* () {
  if (!isWindows(app.candidate.target.os)) return yield* invalid("Requires a Windows package")
  const signatures = [{ ...(yield* verifyWindowsSignature(app.candidate.path, policy, environment)), path: "installer" }]
  const unqualifiedVendors: string[] = []
  const runtime = yield* admittedRuntimeComposition(release, app.candidate.target)
  const fs = yield* FileSystem.FileSystem
  for (const [label, root] of [["application", app.root], ["runtime", runtime.root]] as const) {
    const canonicalRoot = yield* fs.realPath(root)
    const files = yield* nativeInventory(root)
    if (!files.length) return yield* invalid(`No native code in ${label}`)
    for (const required of label === "application" ? [app.executable, app.cli, join(root, "resources", "magnitude-service.exe"),
      join(root, "resources", "desktop-host.node"), join(root, "Uninstall Magnitude.exe")] : [join(root, "bin", "magnitude-inference.exe")]) {
      const canonical = yield* fs.realPath(required).pipe(Effect.mapError(() => invalid(`Required native file is missing or unreadable: ${required}`)))
      if (!files.some(file => file.path === canonical)) return yield* invalid(`Required executable or native bridge is not in the native inventory: ${required}`)
    }
    for (const file of files) {
      if (file.image.format !== "pe") return yield* invalid(`Unexpected native format: ${file.path}`)
      const path = relative(canonicalRoot, file.path).replaceAll("\\", "/")
      const microsoftRuntime = label === "runtime" && /^runtime\/(msvcp140|vcruntime140(_1)?)\.dll$/i.test(path)
      const owned = label === "application" ? ["magnitude.exe", "resources/magnitude.exe", "resources/magnitude-service.exe",
        "resources/desktop-host.node", "uninstall magnitude.exe"].includes(path.toLowerCase())
        : path.toLowerCase() === "bin/magnitude-inference.exe" || /^(runtime|backends)\/(ggml|llama|mtmd)(-.*)?\.dll$/i.test(path)
      // Vendor code retains its vendor signature. Never demand that it impersonate
      // Magnitude, or infer publisher trust merely from a valid arbitrary signer.
      const verification = policy.kind === "production" && !owned && !microsoftRuntime ? { kind: "development" } as const : policy
      signatures.push({ ...(yield* verifyWindowsSignature(file.path, verification, environment, microsoftRuntime)), path: `${label}/${path}` })
      if (policy.kind === "production" && !owned && !microsoftRuntime) unqualifiedVendors.push(`${label}/${path}`)
    }
  }
  if (unqualifiedVendors.length) return yield* new InfrastructureFailure({ operation: "package-trust",
    message: `Third-party native publisher or signed-container provenance is not yet independently verified: ${unqualifiedVendors.join(", ")}` })
  return WindowsPackageTrust.make({ policy, productionTrusted: policy.kind === "production", signatures, runtimeArtifacts: runtime.artifacts })
}))
