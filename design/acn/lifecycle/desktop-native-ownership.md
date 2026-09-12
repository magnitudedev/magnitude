---
applies_to:
  - packages/acn/src/owned-control.ts
  - packages/utils/src/process-groups/**
  - packages/utils/src/windows-native/**
  - packages/utils/src/json-line-channel.ts
  - packages/acn-protocol/src/desktop-control.ts
  - packages/daemon-management/native/**
  - packages/daemon-management/src/desktop-native/**
  - packages/daemon-management/scripts/build-native.ts
  - packages/daemon-management/scripts/*windows*.ps1
  - .github/workflows/desktop-native.yml
---

# Native application ownership

The desktop process owns the service lifetime. A passive kernel lock excludes concurrent desktop
owners without terminating or replacing an unresponsive owner. A contender forwards intent through
local application control or reports bounded unavailability. Lock acquisition is not service readiness.

The lock file lives in a private local user directory. Never unlink or replace it during recovery.
The kernel releases ownership when the owning process exits. Child processes must not inherit the
lock. Explicit release is idempotent and occurs after owned-child shutdown. Filesystem or permission
errors are failures, never evidence of another healthy owner or permission to start a service.
Native lock paths reject embedded NUL before filesystem access. Lock release requires the exact
tagged native capability; plain objects and prototype-derived objects cannot release ownership.
Unix control paths are validated against the platform's UTF-8 socket-path limit before filesystem
changes or client connection attempts. Both lifecycle and login requests reject invalid paths with
an actionable error; they are explicit failures, not missing-owner observations that permit startup.
Windows lock input names a local drive path (including its extended-length form), and the opened
file must be a disk file with one link. Windows native acquisition creates the final private directory
and lock with explicit current-user ownership and protected current-user-only ACLs at creation.
Existing files and directories must satisfy that contract; acquisition does not rewrite unsafe ACLs.
The directory handle requests directory read access and is retained without delete sharing until lock release;
metadata-only access does not establish that sharing protection. Directory reparse points,
null/broad/inherited ACLs, wrong ownership and invalid file types are failures rather than contention.
Windows application pipe names derive from the retained directory's volume GUID and 128-bit file ID,
not its textual path. Read-only client lookup validates the existing directory and derives the same
name without creating state or acquiring service ownership. Only absence at the initial directory
open is absence; permission or identity-query failures remain errors. Resolving an endpoint does not
establish owner presence or service readiness. Clients resolve it anew for each request. Unsupported
drive types are rejected before state creation; a mounted local volume identity is required.
The compiled CLI embeds its target native adapter and loads it lazily for application operations.
Its installation lookup uses that adapter's native known-folder result before locating the desktop;
it cannot depend on an addon found through an environment-guessed desktop path. Passive help and
version commands do not initialize native adapters. Cross-compilation requires the matching native input.
Native Windows acceptance executes the embedded addon and compiled CLI, with a deliberately invalid
LOCALAPPDATA value, and verifies that passive commands create no ownership state. Missing application
startup must report the native installation path without starting an independent service.
Default Windows ownership state lives beneath the current user's native Local AppData known folder,
with separate production and development namespaces. CLI and desktop share this resolution. A
redirected home or caller-supplied LOCALAPPDATA value cannot redirect ownership. Native lookup failure
does not select a guessed fallback. Explicit isolated state overrides still undergo native directory
validation. Chromium profile creation must not pre-create the protected ownership leaf with inherited
permissions; renderer profile data and the ownership directory have distinct creation authority.
Cold Windows application launch requires the caller's assigned interactive window station and its
ordinary desktop. A noninteractive service or SSH session cannot create an unreachable tray owner.
Native inspection failure is not permission to launch. This checks the assigned desktop rather than
the current input desktop, so locking the user session does not itself prohibit background startup.
An existing owner's control endpoint remains usable by headless callers without this launch check.
Direct desktop startup applies the same check before ownership or service creation and exits without
opening a dialog in an inaccessible desktop. Native acceptance covers both console and noninteractive
contexts; environment variables are not evidence of a graphical Windows session.

On Unix, the service leads its own process group and installs a native lifetime-channel watchdog
before application initialization. Loss of the parent channel terminates the owned group even when
JavaScript is blocked. The channel is a blocking pipe/socket, and its retained descriptor is not
inherited across exec. Separately grouped descendants need their own parent-loss protection; leader
exit alone does not prove tree cleanup. A watchdog cannot execute while its whole process is stopped.
A group-signal permission failure is cleared only by proving full group disappearance within the
existing signal grace period. This covers Darwin groups containing exited, unreaped children; a
still-present group retains its permission failure and cannot be declared retired.
Linux process-stat lookup treats ENOENT and ESRCH as process disappearance, including exit between
opening and reading procfs. Other read failures remain observation failures; process disappearance
alone still does not prove process-group retirement.

Transient Unix shell probes also use a native lifetime-bound process group. A bundled helper retains
the group until command output is drained; command exit is separate from group retirement. Parent
death, cancellation, timeout, and output overflow retire the helper and all ordinary descendants,
even when the command ignores termination or its shell has already exited. No probe acquires service
ownership or changes the application environment.

Windows uses a parent-owned unnamed kill-on-close Job Object for children; a Unix watchdog is not
a substitute. Child creation assigns the job atomically through the process-creation attribute list.
Shared host utilities expose scoped job, private-pipe and command-encoding capabilities. The desktop
and ACN each compose their own child owner; ACN does not depend on desktop daemon management.
No suspended-child assignment interval or ordinary-spawn fallback is permitted. Only explicitly
selected I/O handles are inherited; the job handle is never inherited. Root exit is observed through
the retained process handle, while full retirement is proved by the job's active-process count.
Nested jobs and forced-owner-exit cleanup require native Windows execution, not cross-compilation.
Windows named pipes install a protected current-user DACL at creation and reject remote clients.
The first instance refuses an existing endpoint. Native client PID observation fences child admission.
Pending accept/read/write operations retain their buffers through confirmed cancellation; close cannot
race ahead of an operation being issued. Native I/O waits must not occupy the runtime's shared worker
pool and starve writes or shutdown. Node-API objects retain native ownership until all callbacks
finish; final release closes the pipe. Node, Bun, and compiled-service interoperability are separate
native acceptance requirements.
Application-control transports share the same request schemas, framing, dispatch, and login replies.
The Windows listener retains a pending native instance while admitting at most sixteen requests;
completing a request releases its slot and closes its scoped pipe. Shutdown cancels idle acceptance
and reads. Partial native writes cannot interleave separate frames. Reply preservation through server
handle closure is a native acceptance requirement, including when the client has not yet read.
Windows child standard output and error use a parent-created private pipe with explicitly inherited
write handles. Native process creation does not translate CRT descriptors across JavaScript hosts.
Service diagnostics may share one stream; inference uses separate inherited stdin, stdout and stderr
pipes so parent-lifetime EOF, startup records and diagnostics retain their distinct meanings. Both
forms use the same atomic job ownership and retirement rules. A partial stream-open or spawn failure
closes all acquired handles and cannot leave a child outside retained ownership.
The job wrapper retains the process handle before observing its creation identity, and keeps root
exit distinct from the job's active-process count. A forced handle close remains cleanup authority,
not proof that every descendant has exited.
The Windows job owner belongs to the application scope and admits one service job at a time.
Failed, interrupted, or timed-out retirement retains the job and process handles and excludes a
replacement. Retirement releases them only after both zero active members and root exit are observed;
application-scope exit can force release without misreporting that as observed retirement. Observers
cannot take ownership or dispose of the job. Windows process IDs use the same identity type for
retained process observations and native pipe-client admission.
CLI Quit observes the Windows application through a scoped read-only process handle acquired before
requesting shutdown. It waits on that same handle rather than repeatedly resolving a PID, and checks
that the reply identifies the observed application. Observation grants no termination or job rights.
Permission failures are not process absence; cancellation releases observation without killing the
application or its service. Only the desktop owner retires its service tree.
The service validates kill-on-close containment without breakaway in its immediate Windows job
before connecting its private owner pipe. The desktop checks the pipe client's native PID against
the retained child identity before consuming Booted. Windows commands invoke the known executable
directly with CRT argument encoding; the native boundary orders environment keys using Windows
ordinal case-insensitive comparison and rejects duplicate names or premature block terminators.
Platform build and packaged lifecycle acceptance are required before enabling production ownership.

Acceptance includes live contention without takeover, acquisition after owner death without file
replacement, no inherited lock across exec, rejection of unsafe lock files, and parent-loss cleanup
with stalled JavaScript and real descendants. Scope closure must release the retained lock.

The inherited duplex control channel carries only Booted, Start, Health, StoppingObserved, and Shutdown. The child
waits for Start before application initialization; the owner checks Booted against the retained
child handle first. Health is a projection of ACN's existing lifecycle, not an independent model
cache or readiness authority. Frames are schema-validated, bounded, and independent of diagnostic
output. Malformed or lost control is a lifecycle failure, never permission to adopt another process.
The owner acknowledges the final Stopping health after validating its child identity and retaining
its safe detail. ACN waits at most two seconds for this receipt before closing control. Socket write,
end, and close callbacks alone do not prove receipt across process exit. Missing acknowledgement
cannot prevent teardown. The retained detail wins over a racing child-exit notification, including
when exit occurs before the acknowledgement write callback. This private handshake does not change
public RPC or inference contracts; the service and desktop ship as one matched application.

Login startup is an OS-owned preference, independent of the running service. Application control
can read or explicitly change that preference; it replies only after the native adapter finishes.
These requests never dispatch lifecycle intent or create an independent daemon. Errors are typed
and do not masquerade as successful registration. A cold CLI registration request first ensures the
desktop in the background; unregistration then requests full application Quit. Development builds
report login registration unavailable and never register source executables.
Linux desktop entries use the system env executable to exec the absolute application path with
background intent, preserving the environment and process identity without a shell. This permits
percent-containing paths in GLib, which checks the command before expanding desktop-entry escapes.
Native acceptance validates and launches entries with spaces and reserved characters, checking
the exact background argument rather than only comparing generated text.

Service failure presentation uses ACN's safe detail or a concise typed error message. Diagnostic
stacks and stderr remain in logs instead of becoming the ordinary Status label.
Failed child attempts retain bounded diagnostics in logs even when control-channel closure is
observed before process exit; diagnostic visibility cannot depend on which failure wins that race.

The desktop is a clean cutover from independently installed daemon versions. Users download the
new application and stop and disable any previous daemon installation themselves. The application
does not read old coordination databases, inspect or unregister old startup jobs, retire old
processes, transfer login preferences, or persist migration checkpoints. Existing models, settings,
and unrelated files remain untouched by cutover. Only this application's owned children may be stopped.

Before each service spawn, a loopback port check reports an occupied port as an actionable, retryable
failure inside the supervisor. The tray and application control remain available, including Quit.
The check never contacts or adopts the incumbent, and is diagnostic rather than an ownership lock;
the service's own bind remains authoritative if another process races startup. Once the user removes
the conflict, Retry may start a fresh owned service. Acceptance verifies that retries leave an
incumbent alive, stale or malformed coordination files do not gate startup, and no migration state
is created.

Linux tray-host observation belongs to the application scope, independently of service and renderer
lifetime. Subscribe to watcher ownership, host registration, and property changes before the initial
snapshot. Read the registered-host property from the exact unique owner and verify ownership again;
retired-owner replies cannot become current availability. Signals are coalesced invalidations, not
truth. Missing hosts remain observable, and session-bus loss reconnects without periodically
recreating tray objects. Scope closure releases the connection, match rules, reads, and retries.
Host availability proves protocol support, not pixel visibility or user pinning.

Application memory observation is read-only and rooted at the current desktop process. Native sampling runs outside the Electron thread, bounds process and input sizes, and checks creation identity and ancestry across memory reads. A failed member read invalidates the sample. Observation grants no termination authority and cannot create or restart a service. The host serializes sampling; releasing a subscription stops future work after any in-flight native read completes. Acceptance covers real child allocation and retirement, exclusion of unrelated processes, native failure recovery, and visibility-scoped refresh.

Client device identification exposes only manufacturer and product/model from native OS metadata. It requires no elevated privileges or shell processes and does not export serial numbers or UUIDs. Firmware parsing bounds both input and string lengths. Missing or invalid metadata yields unavailable without delaying service readiness or replacing inference-owned hardware facts.
