param([Parameter(Mandatory = $true)][string]$Directory)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

foreach ($name in @('AZURE_SIGNING_ENDPOINT', 'AZURE_SIGNING_ACCOUNT', 'AZURE_SIGNING_CERTIFICATE_PROFILE', 'MAGNITUDE_WINDOWS_PUBLISHER')) {
  if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) { throw "Missing signing configuration: $name" }
}
& az account show --output none
if ($LASTEXITCODE -ne 0) { throw 'Artifact Signing requires an authenticated Azure CLI session.' }
# Acquire the signing audience while the GitHub login assertion is still valid.
# Azure CLI otherwise first requests this token after the native compilation finishes.
& az account get-access-token --resource 'https://codesigning.azure.net' --output none
if ($LASTEXITCODE -ne 0) { throw 'Could not acquire the Artifact Signing access token.' }
$sdk = Join-Path $env:WindowsSdkDir "bin\$($env:WindowsSDKVersion.TrimEnd('\'))\x64\signtool.exe"
if (!(Test-Path -LiteralPath $sdk)) { throw 'The selected Windows SDK is missing x64 SignTool.' }
if ((Get-Item -LiteralPath $sdk).VersionInfo.FileVersionRaw -lt [version]'10.0.22621.755') {
  throw 'Artifact Signing requires Windows SDK SignTool 10.0.22621.755 or newer.'
}

# The signing client is a pinned build dependency, never part of the application payload.
$archive = Join-Path $Directory 'artifact-signing.zip'
Invoke-WebRequest -UseBasicParsing 'https://api.nuget.org/v3-flatcontainer/microsoft.artifactsigning.client/1.0.128/microsoft.artifactsigning.client.1.0.128.nupkg' -OutFile $archive
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne '74bd7d27e6ce1051409c38d9b46bc8df0400ecd643d51ffbf2ac00869061e40b') {
  throw 'Artifact Signing client checksum mismatch.'
}
$client = Join-Path $Directory 'artifact-signing'
Expand-Archive -LiteralPath $archive -DestinationPath $client
$metadata = Join-Path $Directory 'signing-metadata.json'
@{
  Endpoint = $env:AZURE_SIGNING_ENDPOINT
  CodeSigningAccountName = $env:AZURE_SIGNING_ACCOUNT
  CertificateProfileName = $env:AZURE_SIGNING_CERTIFICATE_PROFILE
  ExcludeCredentials = @('EnvironmentCredential', 'WorkloadIdentityCredential', 'ManagedIdentityCredential',
    'SharedTokenCacheCredential', 'VisualStudioCredential', 'VisualStudioCodeCredential',
    'AzurePowerShellCredential', 'AzureDeveloperCliCredential', 'InteractiveBrowserCredential')
} | ConvertTo-Json | Set-Content -LiteralPath $metadata -Encoding UTF8
$env:MAGNITUDE_WINDOWS_SIGNTOOL = $sdk
$env:MAGNITUDE_WINDOWS_SIGNING_DLIB = Join-Path $client 'bin\x64\Azure.CodeSigning.Dlib.dll'
$env:MAGNITUDE_WINDOWS_SIGNING_METADATA = $metadata
