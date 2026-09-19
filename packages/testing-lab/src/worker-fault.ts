import { Context, Effect, Layer, Option, Schema } from "effect"
import { ApplicationIdentity, LabProcessId } from "./application-identity"
import { InfrastructureFailure } from "./domain"
import { command, ProcessExecutor } from "./process"

export const NativeProcessStart = Schema.NonEmptyString.pipe(Schema.brand("NativeProcessStart"))
export const WorkerFaultRequest = Schema.Struct({ owner: ApplicationIdentity, workerPid: LabProcessId, profile: Schema.NonEmptyString })
export const WorkerFaultReceipt = Schema.Struct({ workerPid: LabProcessId, parentPid: LabProcessId,
  parentStart: NativeProcessStart, executable: Schema.NonEmptyString, terminated: Schema.Literal(true) })
export interface WorkerFault {
  readonly crash: (request: typeof WorkerFaultRequest.Type) => Effect.Effect<typeof WorkerFaultReceipt.Type, InfrastructureFailure>
  readonly verifyParent: (receipt: typeof WorkerFaultReceipt.Type) => Effect.Effect<void, InfrastructureFailure>
}
export const WorkerFault = Context.GenericTag<WorkerFault>("@magnitudedev/testing-lab/WorkerFault")
const failure = (message: string) => new InfrastructureFailure({ operation: "worker-fault", message })

// Linux pidfds bind termination to the inspected process even if its numeric PID is reused.
// The helper receives only typed nonsecret data, never interpolated command arguments.
export const linuxWorkerFaultScript = String.raw`
import json,os,select,signal,sys

def inspect(pid):
    root='/proc/'+str(pid)
    with open(root+'/stat') as f: fields=f.read().rpartition(') ')[2].split()
    with open(root+'/status') as f: status=dict(line.split(':',1) for line in f if ':' in line)
    with open(root+'/cmdline','rb') as f: args=[x.decode() for x in f.read(65536).split(b'\0') if x]
    return {'pid':pid,'ppid':int(fields[1]),'start':fields[19],
            'uid':int(status['Uid'].split()[0]),'exe':os.path.realpath(root+'/exe'),'args':args}

def alive(fd):
    poll=select.poll();poll.register(fd,select.POLLIN)
    return not poll.poll(0)

def require(condition,message):
    if not condition: raise RuntimeError(message)

def main():
    request=json.load(sys.stdin)
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux pidfd support is required')
    if request['operation']=='verify-parent':
        receipt=request['receipt']; fd=os.pidfd_open(receipt['parentPid'])
        try:
            parent=inspect(receipt['parentPid'])
            require(alive(fd) and parent['start']==receipt['parentStart'] and parent['exe']==receipt['executable'],
                    'Persistent inference process was replaced or exited')
            print('{}')
        finally: os.close(fd)
        return
    request=request['request'];pid=request['workerPid'];owner=request['owner']
    require(pid not in [owner['applicationPid'],owner['servicePid'],os.getpid()], 'Refusing to terminate an application or service owner')
    workerfd=os.pidfd_open(pid); parentfd=None
    try:
        worker=inspect(pid)
        prefix=os.path.join(os.path.realpath(request['profile']),'releases')+os.sep
        require(worker['uid']==os.getuid() and worker['exe'].startswith(prefix)
                and os.path.basename(worker['exe'])=='magnitude-inference'
                and len(worker['args'])>1 and worker['args'][1]=='inference-worker',
                'Process is not an owned installed inference worker')
        parentfd=os.pidfd_open(worker['ppid']);parent=inspect(worker['ppid'])
        require(parent['exe']==worker['exe'] and parent['uid']==os.getuid()
                and len(parent['args'])>1 and parent['args'][1]=='serve',
                'Worker does not belong to the persistent inference server')
        ancestor=parent;seen=set()
        for _ in range(16):
            require(ancestor['uid']==os.getuid() and ancestor['pid'] not in seen,'Invalid worker ancestry')
            seen.add(ancestor['pid'])
            if ancestor['pid']==owner['servicePid']: break
            require(ancestor['ppid']>1,'Worker is not descended from the owning service')
            ancestor=inspect(ancestor['ppid'])
        else: raise RuntimeError('Worker ancestry exceeds limit')
        require(alive(workerfd) and alive(parentfd),'Worker or inference server exited before fault injection')
        signal.pidfd_send_signal(workerfd,signal.SIGKILL)
        poll=select.poll();poll.register(workerfd,select.POLLIN)
        require(bool(poll.poll(10000)),'Worker did not exit after fault injection')
        require(alive(parentfd),'Fault also terminated the persistent inference server')
        print(json.dumps({'workerPid':pid,'parentPid':parent['pid'],'parentStart':parent['start'],
                          'executable':worker['exe'],'terminated':True}))
    finally:
        os.close(workerfd)
        if parentfd is not None:os.close(parentfd)
try: main()
except Exception as error:
    print('Native worker fault rejected: '+str(error),file=sys.stderr);sys.exit(1)
`

export const linuxWorkerFault = Layer.effect(WorkerFault, Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const invoke = (input: string) => command("/usr/bin/python3", ["-c", linuxWorkerFaultScript], { stdin: Option.some(input), inheritEnv: false,
      env: { PATH: "/usr/bin:/bin", LC_ALL: "C" }, timeoutMs: 20_000, maxOutputBytes: 16_384 }).pipe(
      Effect.provideService(ProcessExecutor, executor), Effect.flatMap(result => result.exitCode === 0 ? Effect.succeed(result.stdout)
        : Effect.fail(failure(result.stderr.trim().slice(-2000)))))
  return {
    crash: request => Schema.encode(Schema.parseJson(Schema.Struct({ operation: Schema.Literal("crash"), request: WorkerFaultRequest })))({ operation: "crash", request }).pipe(
      Effect.flatMap(invoke), Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WorkerFaultReceipt))),
      Effect.mapError(error => failure(error.message))),
    verifyParent: receipt => Schema.encode(Schema.parseJson(Schema.Struct({ operation: Schema.Literal("verify-parent"), receipt: WorkerFaultReceipt })))({ operation: "verify-parent", receipt }).pipe(
      Effect.flatMap(invoke), Effect.asVoid, Effect.mapError(error => failure(error.message))),
  } satisfies WorkerFault
}))

export const nativeWorkerFault = process.platform === "linux" ? linuxWorkerFault : Layer.succeed(WorkerFault, {
  crash: () => Effect.fail(failure("Native worker termination is not yet qualified on this platform")),
  verifyParent: () => Effect.fail(failure("Native worker termination is not yet qualified on this platform")),
})
