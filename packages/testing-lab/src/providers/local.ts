import { FileSystem } from "@effect/platform"
import { DateTime, Effect, Layer, Option, Schema } from "effect"
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path"
import { InfrastructureFailure } from "../domain"
import { LocalMachine, MachineAllocator, MachineTags, WorkerTransport } from "../machines"
import { command, ProcessExecutor } from "../process"

const marker = ".magnitude-lab-lease.json"
const failed = (message: string) => new InfrastructureFailure({ operation: "local-worker", message })
const safeName = (name: string) => /^ml-[a-f0-9]{12}$/.test(name)
export const localAllocator = (directory: string) => Layer.effect(MachineAllocator, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(directory, { recursive: true, mode: 0o700 })
  const base = yield* fs.realPath(directory)
  const read = (root: string) => Effect.gen(function* () {
    if (dirname(resolve(root)) !== base || !safeName(basename(root))) return yield* failed("Local worker path is outside its allocation directory")
    if ((yield* fs.realPath(root)) !== resolve(root)) return yield* failed("Local worker root was replaced by a symlink")
    const tags = yield* fs.readFileString(join(root, marker)).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(MachineTags))))
    return LocalMachine.make({ provider: "local", root, tags })
  }).pipe(Effect.mapError(() => failed("Local worker ownership cannot be verified")))
  return {
    ensure: (lease, target) => Effect.gen(function* () {
      if (lease.provider !== "local" || !safeName(lease.resourceName)) return yield* failed("Invalid local allocation identity")
      if ((target.arch === "arm64" ? "arm64" : "x64") !== process.arch) return yield* failed("Local worker architecture differs from target")
      if ((target.os === "macos" && process.platform !== "darwin") || (target.os === "windows" && process.platform !== "win32") ||
        (!["macos", "windows"].includes(target.os) && process.platform !== "linux")) return yield* failed("Local worker OS family differs from target")
      if (DateTime.toEpochMillis(lease.expiresAt) <= Date.now()) return yield* failed("Local lease already expired")
      const tags = MachineTags.make({ schemaVersion: 1, leaseId: lease.leaseId, runId: lease.runId, expiresAt: lease.expiresAt })
      const root = join(base, lease.resourceName)
      if (!(yield* fs.exists(root))) {
        yield* fs.makeDirectory(root, { mode: 0o700 })
        yield* fs.writeFileString(join(root, marker), yield* Schema.encode(Schema.parseJson(MachineTags))(tags), { flag: "wx", mode: 0o600 })
      }
      const machine = yield* read(root)
      if (!Schema.equivalence(MachineTags)(tags, machine.tags)) return yield* failed("Local directory belongs to another lease")
      return machine
    }).pipe(Effect.mapError(e => failed(e.message))),
    inventory: () => fs.readDirectory(base).pipe(Effect.flatMap(names => Effect.forEach(names.filter(safeName), name => read(join(base, name)))), Effect.mapError(e => failed(e.message))),
    release: machine => Effect.gen(function* () {
      if (machine.provider !== "local") return yield* failed("Wrong allocator for local release")
      // Validate lexical identity even if the resource was already removed.
      if (dirname(resolve(machine.root)) !== base || !safeName(basename(machine.root))) return yield* failed("Refusing to remove an unrelated directory")
      if (!(yield* fs.exists(machine.root))) return
      const actual = yield* read(machine.root)
      if (!Schema.equivalence(MachineTags)(actual.tags, machine.tags)) return yield* failed("Local ownership changed before release")
      yield* fs.remove(machine.root, { recursive: true })
    }).pipe(Effect.mapError(e => failed(e.message))),
  } satisfies MachineAllocator
}))

/** Local tests are trusted developer work, never a sandbox for untrusted CI sources. */
export const localTransport = Layer.effect(WorkerTransport, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* ProcessExecutor
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "SystemRoot", "TEMP", "APPDATA", "LOCALAPPDATA", "DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const verify = (machine: typeof LocalMachine.Type) => Effect.gen(function* () {
    if ((yield* fs.realPath(machine.root)) !== resolve(machine.root)) return yield* failed("Local worker root was replaced")
    const tags = yield* fs.readFileString(join(machine.root, marker)).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(MachineTags))))
    if (!Schema.equivalence(MachineTags)(tags, machine.tags)) return yield* failed("Local worker ownership changed")
  }).pipe(Effect.mapError(e => failed(e.message)))
  const path = (machine: typeof LocalMachine.Type, value: string) => Effect.gen(function* () {
    yield* verify(machine)
    const absolute = resolve(machine.root, value)
    const rel = relative(machine.root, absolute)
    if (!rel || rel === ".." || rel.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`) || isAbsolute(rel)) return yield* failed("Worker transfer path escapes its owned directory")
    // Reject symlink parents; transfers must not follow a candidate-created link outside the lease.
    for (let current = absolute; current !== machine.root; current = dirname(current)) {
      if (Option.isSome(yield* fs.readLink(current).pipe(Effect.option))) return yield* failed("Worker transfer path contains a symlink")
    }
    return absolute
  })
  return {
    execute: (machine, executable, args, timeoutMs) => machine.provider === "local"
      ? verify(machine).pipe(Effect.zipRight(command(executable, args, { cwd: Option.some(machine.root), timeoutMs, inheritEnv: false, env: environment }).pipe(Effect.provideService(ProcessExecutor, executor))))
      : Effect.fail(failed("Wrong local transport")),
    upload: (machine, local, remote) => Effect.gen(function* () {
      if (machine.provider !== "local") return yield* failed("Wrong local transport")
      const destination = yield* path(machine, remote)
      yield* fs.makeDirectory(dirname(destination), { recursive: true, mode: 0o700 })
      yield* fs.copyFile(local, destination)
    }).pipe(Effect.mapError(e => failed(e.message))),
    download: (machine, remote, local) => Effect.gen(function* () {
      if (machine.provider !== "local") return yield* failed("Wrong local transport")
      const source = yield* path(machine, remote)
      yield* fs.copyFile(source, local)
    }).pipe(Effect.mapError(e => failed(e.message))),
  } satisfies WorkerTransport
}))
