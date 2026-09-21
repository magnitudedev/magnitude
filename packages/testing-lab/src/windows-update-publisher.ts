import { Effect, Schema } from "effect"
import { X509Certificate } from "node:crypto"
import { InfrastructureFailure } from "./domain"
import { checkedCommand } from "./process"

export const WindowsUpdatePublisher = Schema.Struct({
  thumbprint: Schema.String.pipe(Schema.pattern(/^[A-F0-9]{40}$/)),
  certificate: Schema.String.pipe(Schema.minLength(1), Schema.maxLength(16 * 1024)),
})
const fail = () => new InfrastructureFailure({ operation: "windows-update-publisher", message: "Invalid isolated Windows update publisher" })
const run = (script: string, environment: Readonly<Record<string, string>> = {}) => checkedCommand("powershell.exe",
  ["-NoProfile", "-NonInteractive", "-Command", script], { env: environment, timeoutMs: 30_000, maxOutputBytes: 32 * 1024 })

export const validateWindowsUpdatePublisher = (publisher: typeof WindowsUpdatePublisher.Type) => Effect.try({
  try: () => {
    const certificate = new X509Certificate(Buffer.from(publisher.certificate, "base64"))
    const subject = certificate.toLegacyObject().subject
    if (certificate.fingerprint.replaceAll(":", "") !== publisher.thumbprint
      || subject.CN !== "Magnitude Update Acceptance" || subject.O !== "Magnitude Update Acceptance"
      || Object.keys(subject).length !== 2
      || certificate.issuer !== certificate.subject || !certificate.verify(certificate.publicKey)
      || Date.parse(certificate.validFrom) > Date.now() || Date.parse(certificate.validTo) <= Date.now()) throw new Error("Invalid certificate")
    return publisher
  }, catch: fail,
})

/** Private key stays on the disposable producer; only the public DER is transferred. */
export const createWindowsUpdatePublisher = Effect.acquireRelease(
  run(String.raw`
$ErrorActionPreference = 'Stop'
$certificate = $null
try {
  $certificate = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=Magnitude Update Acceptance, O=Magnitude Update Acceptance' -CertStoreLocation Cert:\CurrentUser\My -KeyAlgorithm RSA -KeyLength 2048 -HashAlgorithm SHA256 -NotAfter (Get-Date).AddDays(2)
  $store = [Security.Cryptography.X509Certificates.X509Store]::new('Root','LocalMachine')
  try { $store.Open('ReadWrite'); $store.Add([Security.Cryptography.X509Certificates.X509Certificate2]::new($certificate.RawData)) } finally { $store.Close() }
  @{ thumbprint = $certificate.Thumbprint; certificate = [Convert]::ToBase64String($certificate.RawData) } | ConvertTo-Json -Compress
} catch {
  if ($certificate) {
    Remove-Item -LiteralPath "Cert:\LocalMachine\Root\$($certificate.Thumbprint)" -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "Cert:\CurrentUser\My\$($certificate.Thumbprint)" -DeleteKey -ErrorAction SilentlyContinue
  }
  throw
}
`).pipe(Effect.flatMap(result => Schema.decodeUnknown(Schema.parseJson(WindowsUpdatePublisher))(result.stdout))),
  publisher => removePublisher(publisher.thumbprint, true).pipe(Effect.orDie),
).pipe(Effect.flatMap(validateWindowsUpdatePublisher))

const removePublisher = (thumbprint: string, privateKey: boolean) => run(String.raw`
$ErrorActionPreference = 'Stop'
foreach ($storeName in ($env:LAB_PUBLISHER_STORES -split ',')) {
  $path = "Cert:\$storeName\$env:LAB_PUBLISHER_THUMBPRINT"
  if (Test-Path -LiteralPath $path) {
    if ($storeName -eq 'CurrentUser\My') { Remove-Item -LiteralPath $path -Force -DeleteKey } else { Remove-Item -LiteralPath $path -Force }
  }
}
`, { LAB_PUBLISHER_THUMBPRINT: thumbprint, LAB_PUBLISHER_STORES: privateKey ? "LocalMachine\\Root,CurrentUser\\My" : "LocalMachine\\Root" })

/** Authenticode trust is scoped to the disposable Windows VM and removed after the update journey. */
export const trustWindowsUpdatePublisher = (publisher: typeof WindowsUpdatePublisher.Type) => Effect.gen(function* () {
  yield* validateWindowsUpdatePublisher(publisher)
  yield* Effect.acquireRelease(run(String.raw`
$ErrorActionPreference = 'Stop'
$certificate = [Security.Cryptography.X509Certificates.X509Certificate2]::new([Convert]::FromBase64String($env:LAB_PUBLISHER_CERTIFICATE))
if ($certificate.Thumbprint -ne $env:LAB_PUBLISHER_THUMBPRINT -or $certificate.HasPrivateKey) { throw 'Publisher certificate mismatch' }
if (Test-Path -LiteralPath "Cert:\LocalMachine\Root\$($certificate.Thumbprint)") { throw 'Publisher trust already exists; refusing to claim ownership' }
$store = [Security.Cryptography.X509Certificates.X509Store]::new('Root','LocalMachine')
  try { $store.Open('ReadWrite'); $store.Add([Security.Cryptography.X509Certificates.X509Certificate2]::new($certificate.RawData)) } finally { $store.Close() }
`, { LAB_PUBLISHER_CERTIFICATE: publisher.certificate, LAB_PUBLISHER_THUMBPRINT: publisher.thumbprint }),
    () => removePublisher(publisher.thumbprint, false).pipe(Effect.orDie))
})
