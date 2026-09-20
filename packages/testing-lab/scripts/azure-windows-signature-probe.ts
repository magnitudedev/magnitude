import { Fence } from "../src/lease"
import { WorkId } from "../src/work-identity"
import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Context, DateTime, Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { findTarget } from "../src/catalog"
import { InfrastructureFailure, LeaseId, RunId, TargetId } from "../src/domain"
import { Allocating } from "../src/lease"
import { MachineAllocator } from "../src/machines"
import { azureAllocator, AzureConfig } from "../src/providers/azure"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { windowsSignatureScript, WindowsSignatureObservation } from "../src/suites/windows-package-trust"

// Native inspection diagnostic only. A Server image cannot qualify a Windows client target.
const failure = (message: string) => new InfrastructureFailure({ operation: "windows-signature-probe", message })
BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("LAB_AZURE_PROBE_ROOT")
  const config = yield* fs.readFileString(yield* Config.string("LAB_AZURE_CONFIG")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(AzureConfig))))
  const target = yield* findTarget(TargetId.make(yield* Config.string("LAB_AZURE_TARGET")))
  if (target.os !== "windows" || target.backend !== "cpu" || !["intel", "amd"].includes(target.hardware)) return yield* failure("Diagnostic requires a Windows CPU allocation")
  yield* fs.makeDirectory(root, { recursive: true })
  const allocator = Context.get(yield* Layer.build(azureAllocator(config)), MachineAllocator)
  const lease = new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: target.id, workId: WorkId.make(`test:${target.id}`), workFence: Fence.make(1),
    provider: "azure", resourceName: `ml-${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`, expiresAt: DateTime.unsafeMake(Date.now() + 60 * 60_000) })
  yield* fs.writeFileString(join(root, "lease.json"), yield* Schema.encode(Schema.parseJson(Allocating))(lease), { flag: "wx" })
  const cleanupErrors: string[] = []
  const result = yield* Effect.gen(function* () {
    const machine = yield* allocator.ensure(lease, target)
    if (machine.provider !== "azure") return yield* failure("Allocator returned another provider")
    const script = join(root, "inspect.ps1")
    yield* fs.writeFileString(script, String.raw`
$ErrorActionPreference = 'Stop'
$inspect = {
${windowsSignatureScript}
}
$root = Join-Path ([IO.Path]::GetTempPath()) ('magnitude-signature-' + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $root | Out-Null
try {
  $signed = Join-Path $root "signed fixture's [copy].exe"
  # Read-only fixture from Microsoft's official redistributable download; never execute it.
  # Pin bytes as well as publisher so a moving download cannot silently alter the test.
  $source = Join-Path $root 'VC_redist.x64.exe'
  Invoke-WebRequest -UseBasicParsing -Uri 'https://download.visualstudio.microsoft.com/download/pr/bd1c8d9d-ba95-4eee-bc6e-df1fcc876373/CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B/VC_redist.x64.exe' -OutFile $source -TimeoutSec 120
  $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
  if ($hash -cne 'CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B') { throw 'Native signature fixture digest changed' }
  Copy-Item -LiteralPath $source -Destination $signed
  $env:LAB_SIGNATURE_PATH = $signed
  $valid = (& $inspect) | ConvertFrom-Json
  if ($valid.status -ne 'Valid' -or $valid.signatureType -ne 'Authenticode' -or $valid.publisher -cne 'Microsoft Corporation') { throw 'Native fixture lacks the expected embedded Microsoft signature' }
  $unsigned = Join-Path $root 'unsigned.exe'
  Add-Type -TypeDefinition 'public class SignatureFixture { public static void Main() {} }' -Language CSharp -OutputAssembly $unsigned -OutputType ConsoleApplication
  $env:LAB_SIGNATURE_PATH = $unsigned
  $absent = (& $inspect) | ConvertFrom-Json
  $bytes = [IO.File]::ReadAllBytes($signed)
  $pe = [BitConverter]::ToInt32($bytes, 60)
  $section = $pe + 24 + [BitConverter]::ToUInt16($bytes, $pe + 20)
  $length = [BitConverter]::ToUInt32($bytes, $section + 16)
  $offset = [BitConverter]::ToUInt32($bytes, $section + 20)
  if ($length -lt 17 -or $offset -lt ($section + 40) -or ($offset + 17) -gt $bytes.Length) { throw 'Invalid fixture section bounds' }
  $bytes[$offset + 16] = $bytes[$offset + 16] -bxor 1
  [IO.File]::WriteAllBytes($signed, $bytes)
  $env:LAB_SIGNATURE_PATH = $signed
  $corrupt = (& $inspect) | ConvertFrom-Json
  $os = Get-CimInstance Win32_OperatingSystem
  @{ os = $os.Caption; build = $os.BuildNumber; fixture = 'VC_redist.x64.exe'; fixtureSha256 = $hash; signed = $valid; unsigned = $absent; corrupt = $corrupt } | ConvertTo-Json -Depth 4 -Compress
} finally { Remove-Item -LiteralPath $root -Recurse -Force }
`)
    const response = yield* checkedCommand(config.executable, ["vm", "run-command", "invoke", "--subscription", config.subscription,
      "--resource-group", config.resourceGroup, "--name", machine.name, "--command-id", "RunPowerShellScript", "--scripts", `@${script}`, "--output", "json", "--only-show-errors"], { timeoutMs: 600_000 })
    yield* fs.writeFileString(join(root, "run-command.json"), response.stdout)
    const body = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ value: Schema.Array(Schema.Struct({ code: Schema.String, message: Schema.String })) })))(response.stdout)
    const output = body.value.find(item => item.code === "ComponentStatus/StdOut/succeeded")?.message
    if (!output) return yield* failure("Native inspection returned no stdout report; see run-command.json")
    const Observation = Schema.Struct({ os: Schema.String, build: Schema.String, fixture: Schema.String, fixtureSha256: Schema.String, signed: WindowsSignatureObservation,
      unsigned: WindowsSignatureObservation, corrupt: WindowsSignatureObservation })
    const observed = yield* Schema.decodeUnknown(Schema.parseJson(Observation))(output.trim())
    yield* fs.writeFileString(join(root, "observation.json"), yield* Schema.encode(Schema.parseJson(Observation))(observed))
    if (observed.signed.status !== "Valid" || observed.signed.signatureType !== "Authenticode" || observed.unsigned.status !== "NotSigned" || observed.corrupt.status !== "HashMismatch") {
      return yield* failure("Native signed, unsigned or corrupt observations differed from expected Authenticode behavior")
    }
  }).pipe(Effect.ensuring(allocator.inventory().pipe(Effect.flatMap(machines => Effect.forEach(machines.filter(machine => machine.tags.leaseId === lease.leaseId), machine => allocator.release(machine))),
    Effect.catchAll(error => Effect.sync(() => { cleanupErrors.push(error.message) })))), Effect.either)
  yield* fs.writeFileString(join(root, "report.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ passed: result._tag === "Right" && cleanupErrors.length === 0,
    detail: result._tag === "Right" ? "Native Authenticode inspection verified signed, unsigned and changed bytes; no Windows client or application acceptance claimed" : String(result.left), cleanupErrors }))
  if (result._tag === "Left") return yield* result.left
  if (cleanupErrors.length) return yield* failure("Cleanup failed; inspect the owned lease")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
