import { Effect, Option, Schema } from "effect"
import { ApplicationIdentity, LabProcessId } from "../application-identity"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { command } from "../process"

const ProcessBirth = Schema.NonEmptyString.pipe(Schema.brand("RemovalProcessBirth"))
export const RemovalProcesses = Schema.NonEmptyArray(Schema.Struct({ pid: LabProcessId, birth: ProcessBirth, executable: Schema.String }))
const Request = Schema.Union(Schema.TaggedStruct("Capture", { owner: ApplicationIdentity }), Schema.TaggedStruct("Verify", { processes: RemovalProcesses }))
// The native boundary records birth identity so PID reuse cannot create a false leak report.
const inspectUnix = String.raw`
import json,os,sys
request=json.load(sys.stdin)
def read_linux(pid,allow_inaccessible=False):
    root='/proc/'+str(pid)
    try:
        fields=open(root+'/stat').read().rpartition(') ')[2].split()
        result={'pid':pid,'parent':int(fields[1]),'birth':fields[19],'executable':os.readlink(root+'/exe'),'uid':os.stat(root).st_uid}
        if open(root+'/stat').read().rpartition(') ')[2].split()[19]!=result['birth']:return None
        return result
    except (FileNotFoundError,ProcessLookupError):return None
    except PermissionError:
        if allow_inaccessible:return None
        raise
if sys.platform=='darwin':
    import ctypes,errno
    # Public proc_bsdinfo from the macOS SDK: creation time includes microseconds.
    class BsdInfo(ctypes.Structure):
        _fields_=[(name,ctypes.c_uint32) for name in ['flags','status','xstatus','pid','parent','uid','gid','ruid','rgid','svuid','svgid','reserved']]+[('comm',ctypes.c_char*16),('name',ctypes.c_char*32)]+[(name,ctypes.c_uint32) for name in ['nfiles','pgid','jobc','tdev','tpgid']]+[('nice',ctypes.c_int32),('seconds',ctypes.c_uint64),('microseconds',ctypes.c_uint64)]
    lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    lib.proc_pidinfo.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int]
    lib.proc_pidpath.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
    lib.proc_listallpids.argtypes=[ctypes.c_void_p,ctypes.c_int]
    def read(pid,allow_inaccessible=False):
        info=BsdInfo()
        count=lib.proc_pidinfo(pid,3,0,ctypes.byref(info),ctypes.sizeof(info))
        if count!=ctypes.sizeof(info):
            error=ctypes.get_errno()
            if error in (errno.ESRCH,errno.ENOENT) or (allow_inaccessible and error in (errno.EPERM,errno.EACCES)):return None
            raise RuntimeError('Cannot inspect native process birth identity')
        if info.status==5:return None # A zombie has exited and cannot run.
        if allow_inaccessible and info.uid!=os.getuid():return None
        path=ctypes.create_string_buffer(4096)
        if lib.proc_pidpath(pid,path,len(path))<=0:
            if ctypes.get_errno() in (errno.ESRCH,errno.ENOENT):return None
            raise RuntimeError('Cannot inspect native process executable')
        after=BsdInfo()
        if lib.proc_pidinfo(pid,3,0,ctypes.byref(after),ctypes.sizeof(after))!=ctypes.sizeof(after):
            if ctypes.get_errno() in (errno.ESRCH,errno.ENOENT):return None
            raise RuntimeError('Cannot recheck native process identity')
        if (info.seconds,info.microseconds)!=(after.seconds,after.microseconds):return None
        return {'pid':pid,'parent':info.parent,'birth':str(info.seconds)+':'+str(info.microseconds),'executable':path.value.decode(),'uid':info.uid}
    def pids():
        values=(ctypes.c_int*65536)()
        count=lib.proc_listallpids(values,ctypes.sizeof(values))
        if count<=0 or count>=len(values):raise RuntimeError('Native process inventory failed or exceeded bound')
        return values[:count]
else:
    read=read_linux
    def pids():return [int(x) for x in os.listdir('/proc') if x.isdecimal()]
if request['_tag']=='Capture':
    roots={request['owner']['applicationPid'],request['owner']['servicePid']}
    seen={pid:read(pid,True) for pid in pids()}
    if any(not seen.get(pid) or seen[pid]['uid']!=os.getuid() for pid in roots):raise RuntimeError('Owned app and service must both be live')
    owned=set(roots)
    for _ in range(64):
        children={pid for pid,value in seen.items() if value and value['parent'] in owned and value['uid']==os.getuid()}
        following=owned|children
        if following==owned:break
        owned=following
        if len(owned)>1024:raise RuntimeError('Owned process tree exceeded bound')
    else:raise RuntimeError('Owned process tree exceeded depth')
    print(json.dumps([{key:seen[pid][key] for key in ['pid','birth','executable']} for pid in sorted(owned)]))
else:
    live=[]
    for expected in request['processes']:
        current=read(expected['pid'])
        if current and current['birth']==expected['birth']:live.append(expected['pid'])
    print(json.dumps(live))
`
const inspectWindows = String.raw`
$ErrorActionPreference = 'Stop'
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
function Read-Process([int]$id) {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $id"
  if ($null -eq $process) { return $null }
  if ($null -eq $process.CreationDate) { throw 'Missing native process creation time' }
  return $process
}
if ($request._tag -eq 'Capture') {
  $roots = @([int]$request.owner.applicationPid, [int]$request.owner.servicePid) | Select-Object -Unique
  $seen = @{}
  foreach ($process in @(Get-CimInstance Win32_Process)) { $seen[[int]$process.ProcessId] = $process }
  $owned = [Collections.Generic.HashSet[int]]::new()
  foreach ($id in $roots) {
    $process = Read-Process $id
    if ($null -eq $process) { throw 'Owned app and service must both be live' }
    $identity = Invoke-CimMethod -InputObject $process -MethodName GetOwnerSid
    if ($identity.ReturnValue -ne 0 -or $identity.Sid -ne $currentSid) { throw 'Process belongs to another user' }
    $seen[$id] = $process
    [void]$owned.Add($id)
  }
  for ($depth = 0; $depth -lt 64; $depth++) {
    $added = $false
    foreach ($id in @($seen.Keys)) {
      $process = $seen[$id]
      if (-not $owned.Contains($id) -and $owned.Contains([int]$process.ParentProcessId) -and
          $null -ne $process.CreationDate -and $process.CreationDate -ge $seen[[int]$process.ParentProcessId].CreationDate) {
        $identity = Invoke-CimMethod -InputObject $process -MethodName GetOwnerSid
        if ($identity.ReturnValue -eq 0 -and $identity.Sid -eq $currentSid) { [void]$owned.Add($id); $added = $true }
      }
    }
    if ($owned.Count -gt 1024) { throw 'Owned process tree exceeded bound' }
    if (-not $added) { break }
  }
  if ($depth -eq 64) { throw 'Owned process tree exceeded depth' }
  $result = @($owned | Sort-Object | ForEach-Object {
    $process = $seen[$_]
    if (-not $process.ExecutablePath -or $null -eq $process.CreationDate) { throw 'Missing native process identity' }
    @{pid=[int]$process.ProcessId; birth=$process.CreationDate.ToUniversalTime().Ticks.ToString(); executable=$process.ExecutablePath}
  })
} else {
  $result = @(foreach ($expected in $request.processes) {
    $process = Read-Process ([int]$expected.pid)
    if ($null -ne $process -and $process.CreationDate.ToUniversalTime().Ticks.ToString() -eq $expected.birth) { [int]$expected.pid }
  })
}
ConvertTo-Json -InputObject @($result) -Compress -Depth 5
`
const invoke = (request: typeof Request.Type) => Effect.gen(function* () {
  if (!["linux", "darwin", "win32"].includes(process.platform)) return yield* new InfrastructureFailure({ operation: "uninstall-processes", message: "Unsupported native process inspection platform" })
  const result = yield* command(process.platform === "win32" ? "powershell.exe" : "python3", process.platform === "win32"
    ? ["-NoProfile", "-NonInteractive", "-Command", inspectWindows] : ["-c", inspectUnix],
  { stdin: Option.some(yield* Schema.encode(Schema.parseJson(Request))(request)), timeoutMs: 30_000 })
  if (result.exitCode !== 0) return yield* new InfrastructureFailure({ operation: "uninstall-processes", message: "Could not inspect the owned process tree" })
  return result.stdout
})
export const captureRemovalProcesses = (owner: ApplicationIdentity) => invoke({ _tag: "Capture", owner }).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(RemovalProcesses))))
export const verifyRemovalProcesses = (processes: typeof RemovalProcesses.Type) => Effect.gen(function* () {
  const live = yield* invoke({ _tag: "Verify", processes }).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Array(LabProcessId)))))
  if (live.length) return yield* new AssertionFailure({ message: `Owned application processes survived uninstall: ${live.join(", ")}` })
  return processes
})
