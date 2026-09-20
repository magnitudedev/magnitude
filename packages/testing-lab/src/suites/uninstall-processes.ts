import { Effect, Option, Schema } from "effect"
import { ApplicationIdentity, LabProcessId } from "../application-identity"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { command } from "../process"

const ProcessBirth = Schema.NonEmptyString.pipe(Schema.brand("RemovalProcessBirth"))
export const RemovalProcesses = Schema.NonEmptyArray(Schema.Struct({ pid: LabProcessId, birth: ProcessBirth, executable: Schema.String }))
const Request = Schema.Union(Schema.TaggedStruct("Capture", { owner: ApplicationIdentity }), Schema.TaggedStruct("Verify", { processes: RemovalProcesses }))
// The native boundary records birth identity so PID reuse cannot create a false leak report.
const inspect = String.raw`
import json,os,sys
request=json.load(sys.stdin)
def read(pid,allow_inaccessible=False):
    root='/proc/'+str(pid)
    try:
        fields=open(root+'/stat').read().rpartition(') ')[2].split()
        return {'pid':pid,'parent':int(fields[1]),'birth':fields[19],'executable':os.readlink(root+'/exe'),'uid':os.stat(root).st_uid}
    except (FileNotFoundError,ProcessLookupError):return None
    except PermissionError:
        if allow_inaccessible:return None
        raise
if request['_tag']=='Capture':
    roots={request['owner']['applicationPid'],request['owner']['servicePid']}
    seen={int(x):read(int(x),True) for x in os.listdir('/proc') if x.isdecimal()}
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
const invoke = (request: typeof Request.Type) => Effect.gen(function* () {
  if (process.platform !== "linux") return yield* new InfrastructureFailure({ operation: "uninstall-processes", message: "Native process-tree removal verification is currently qualified for Linux only" })
  const result = yield* command("python3", ["-c", inspect], { stdin: Option.some(yield* Schema.encode(Schema.parseJson(Request))(request)), timeoutMs: 30_000 })
  if (result.exitCode !== 0) return yield* new InfrastructureFailure({ operation: "uninstall-processes", message: "Could not inspect the owned process tree" })
  return result.stdout
})
export const captureRemovalProcesses = (owner: ApplicationIdentity) => invoke({ _tag: "Capture", owner }).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(RemovalProcesses))))
export const verifyRemovalProcesses = (processes: typeof RemovalProcesses.Type) => Effect.gen(function* () {
  const live = yield* invoke({ _tag: "Verify", processes }).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Array(LabProcessId)))))
  if (live.length) return yield* new AssertionFailure({ message: `Owned application processes survived uninstall: ${live.join(", ")}` })
  return processes
})
