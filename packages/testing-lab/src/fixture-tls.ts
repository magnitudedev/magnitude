import { FileSystem } from "@effect/platform"
import { Effect } from "effect"
import { join } from "node:path"
import { checkedCommand } from "./process"

/** Ephemeral loopback TLS material; neither implementation changes system trust. */
export const createFixtureTls = (directory: string, platform: NodeJS.Platform = process.platform) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const certificate = join(directory, "certificate.pem"), key = join(directory, "tls-key.pem")
  if (platform === "win32") {
    // PowerShell 7 is already a pinned worker prerequisite. Its .NET certificate
    // request creates an exportable key entirely in memory, without a store entry.
    yield* checkedCommand("pwsh.exe", ["-NoProfile", "-NonInteractive", "-Command", String.raw`
$ErrorActionPreference = 'Stop'
$rsa = [Security.Cryptography.RSA]::Create(2048)
$certificate = $null
try {
  $request = [Security.Cryptography.X509Certificates.CertificateRequest]::new('CN=Magnitude isolated update fixture', $rsa, [Security.Cryptography.HashAlgorithmName]::SHA256, [Security.Cryptography.RSASignaturePadding]::Pkcs1)
  $request.CertificateExtensions.Add([Security.Cryptography.X509Certificates.X509BasicConstraintsExtension]::new($true, $false, 0, $true))
  $usage = [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature -bor [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyEncipherment -bor [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyCertSign
  $request.CertificateExtensions.Add([Security.Cryptography.X509Certificates.X509KeyUsageExtension]::new($usage, $true))
  $purposes = [Security.Cryptography.OidCollection]::new()
  [void]$purposes.Add([Security.Cryptography.Oid]::new('1.3.6.1.5.5.7.3.1'))
  $request.CertificateExtensions.Add([Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]::new($purposes, $false))
  $names = [Security.Cryptography.X509Certificates.SubjectAlternativeNameBuilder]::new()
  $names.AddIpAddress([Net.IPAddress]::Loopback)
  $names.AddDnsName('localhost')
  $request.CertificateExtensions.Add($names.Build())
  $certificate = $request.CreateSelfSigned([DateTimeOffset]::UtcNow.AddMinutes(-1), [DateTimeOffset]::UtcNow.AddDays(1))
  [IO.File]::WriteAllText($env:LAB_FIXTURE_CERTIFICATE, $certificate.ExportCertificatePem(), [Text.UTF8Encoding]::new($false))
  [IO.File]::WriteAllText($env:LAB_FIXTURE_TLS_KEY, $rsa.ExportPkcs8PrivateKeyPem(), [Text.UTF8Encoding]::new($false))
} finally {
  if ($certificate) { $certificate.Dispose() }
  $rsa.Dispose()
}
`], { env: { LAB_FIXTURE_CERTIFICATE: certificate, LAB_FIXTURE_TLS_KEY: key }, timeoutMs: 30_000, maxOutputBytes: 64 * 1024 })
  } else {
    const config = join(directory, "openssl.cnf")
    yield* fs.writeFileString(config, `[req]\nprompt = no\ndistinguished_name = subject\nx509_extensions = extensions\n[subject]\nCN = Magnitude isolated update fixture\n[extensions]\nbasicConstraints = critical,CA:TRUE\nkeyUsage = critical,digitalSignature,keyEncipherment,keyCertSign\nextendedKeyUsage = serverAuth\nsubjectAltName = IP:127.0.0.1,DNS:localhost\n`, { mode: 0o600 })
    yield* checkedCommand("openssl", ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-config", config,
      "-keyout", key, "-out", certificate], { timeoutMs: 30_000, maxOutputBytes: 64 * 1024 })
  }
  yield* fs.chmod(key, 0o600)
})
