import { Effect, Redacted, Schema } from "effect"
import { InfrastructureFailure } from "../domain"
import { InitializationDownload } from "./linux-initialization"

/** Driver installation is privileged provider preparation, never submitted source. */
export const NvidiaPreparation = Schema.Struct({
  model: Schema.Literal("a10", "rtx-pro-6000"),
  version: Schema.String.pipe(Schema.pattern(/^\d{3}\.\d{2,3}(?:\.\d{2})?$/)),
  download: InitializationDownload.pipe(Schema.filter(item => {
    const url = new URL(Redacted.value(item.url))
    return url.hostname === "download.microsoft.com" && !url.search && url.pathname.startsWith("/download/")
  })),
})
export type NvidiaPreparation = typeof NvidiaPreparation.Type

export const nvidiaPreparationScript = (config: NvidiaPreparation, os: "Linux" | "Windows") => Effect.gen(function* () {
  const json = yield* Schema.encode(Schema.parseJson(NvidiaPreparation))(config)
  const encoded = Buffer.from(json).toString("base64")
  if (os === "Windows") return String.raw`$ErrorActionPreference='Stop'
if(-not [Security.Principal.WindowsIdentity]::GetCurrent().IsSystem){throw 'GPU preparation requires SYSTEM'}
$config=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${encoded}')) | ConvertFrom-Json
$root='C:\MagnitudeLab\Preparation\Nvidia'
New-Item -ItemType Directory -Path $root -Force | Out-Null
$installer=Join-Path $root 'driver.exe'
try {
  & curl.exe --fail --silent --show-error --proto '=https' --max-time 600 --max-filesize $config.download.bytes --output $installer $config.download.url
  if($LASTEXITCODE -ne 0){throw 'GPU driver download failed'}
  if((Get-Item -LiteralPath $installer).Length -ne $config.download.bytes -or (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant() -ne $config.download.sha256){throw 'GPU driver integrity mismatch'}
  $signature=Get-AuthenticodeSignature -LiteralPath $installer
  if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'NVIDIA Corporation'){throw 'GPU driver publisher verification failed'}
  $process=Start-Process -FilePath $installer -ArgumentList @('-s','-noreboot') -Wait -PassThru
  if($process.ExitCode -notin @(0,3010)){throw "GPU driver installer exited $($process.ExitCode)"}
} finally {Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue}
# Device readiness is checked after the allocator's single desktop reboot.
`
  return String.raw`set -eu
python3 - <<'LAB_GPU'
import base64,hashlib,json,os,pathlib,platform,subprocess,tempfile,urllib.request
config=json.loads(base64.b64decode('${encoded}'))
if os.geteuid()!=0 or platform.machine()!='x86_64':raise SystemExit('GPU preparation requires Linux x64 root')
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
opener=urllib.request.build_opener(NoRedirect)
with tempfile.TemporaryDirectory(prefix='magnitude-nvidia-') as root:
 installer=pathlib.Path(root)/'driver.run'
 digest=hashlib.sha256();total=0
 with opener.open(config['download']['url'],timeout=120) as response,installer.open('xb') as output:
  for chunk in iter(lambda:response.read(1024*1024),b''):
   total+=len(chunk)
   if total>config['download']['bytes']:raise SystemExit('GPU driver exceeds pinned length')
   digest.update(chunk);output.write(chunk)
 if total!=config['download']['bytes'] or digest.hexdigest()!=config['download']['sha256']:raise SystemExit('GPU driver integrity mismatch')
 kernel=platform.release()
 # Headers must match the running image kernel, not the repository's newest kernel.
 subprocess.run(['apt-get','install','-y','build-essential','linux-headers-'+kernel],check=True,env={**os.environ,'DEBIAN_FRONTEND':'noninteractive','NEEDRESTART_MODE':'l'})
 args=['/bin/sh',str(installer),'--silent','--no-questions','--no-nouveau-check']
 if config['model']=='rtx-pro-6000':args+=['-M','open']
 subprocess.run(args,check=True)
 subprocess.run(['modprobe','nvidia'],check=True)
LAB_GPU
`
}).pipe(Effect.mapError(() => new InfrastructureFailure({ operation: "nvidia-preparation", message: "Cannot encode the pinned GPU driver recipe" })))

/** Installation success cannot substitute for native model and driver observation. */
export const nvidiaReadinessScript = (config: NvidiaPreparation, os: "Linux" | "Windows") => {
  const pattern = config.model === "a10" ? "(?:^| )A10(?:-| |$)" : "RTX PRO 6000"
  if (os === "Windows") return String.raw`$ErrorActionPreference='Stop'
$smi=Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if(-not $smi){$path='C:\Windows\System32\nvidia-smi.exe'}else{$path=$smi.Source}
$rows=@(& $path --query-gpu=name,driver_version --format=csv,noheader)
if($LASTEXITCODE -ne 0 -or $rows.Count -ne 1){throw 'Expected one usable NVIDIA GPU'}
$fields=$rows[0].Split(',').Trim()
if($fields.Count -ne 2 -or $fields[0] -notmatch '${pattern}' -or $fields[1] -ne '${config.version}'){throw 'GPU model or driver differs from the admitted recipe'}
Write-Output $rows[0]
`
  return String.raw`set -eu
python3 - <<'LAB_GPU_READY'
import re,subprocess
rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip().splitlines()
if len(rows)!=1:raise SystemExit('Expected one usable NVIDIA GPU')
fields=[field.strip() for field in rows[0].split(',')]
if len(fields)!=2 or not re.search('${pattern}',fields[0]) or fields[1]!='${config.version}':raise SystemExit('GPU model or driver differs from the admitted recipe')
print(rows[0])
LAB_GPU_READY
`
}
