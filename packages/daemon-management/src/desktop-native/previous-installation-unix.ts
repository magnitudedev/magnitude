import { realpath, readlink, stat } from "node:fs/promises"
import { basename, isAbsolute, join, relative } from "node:path"
import { Effect, Option, Schema } from "effect"
import { ProcessGroupController, ProcessGroupSchema } from "@magnitudedev/utils/process-groups"
import { SqliteDriver } from "../sqlite-driver"
import { readLegacyOwner, type LegacyOwner } from "./legacy-owner"
import { captureLegacyTree, LegacyProcessTable } from "./legacy-tree"
import { LegacyStartup } from "./legacy-startup"
import { LegacyStartupCommands, LegacyStartupFailed } from "./legacy-startup-command"
import { PreviousInstallation, PreviousInstallationFailed } from "./previous-installation"

const failed = (error: { readonly message: string }) => new PreviousInstallationFailed({ message: error.message })
const inside = (root: string, path: string) => { const part = relative(root, path); return part !== "" && !part.startsWith("..") && !isAbsolute(part) }

export const makeUnixPreviousInstallation = (options: { readonly dataDirectory: string; readonly home: string }) => Effect.gen(function* () {
  const processes = yield* ProcessGroupController
  const table = yield* LegacyProcessTable
  const commands = yield* LegacyStartupCommands
  const sqlite = yield* SqliteDriver
  const startup = yield* LegacyStartup
  const read = readLegacyOwner(options.dataDirectory).pipe(Effect.provideService(SqliteDriver, sqlite))
  const capture = (owner: LegacyOwner) => captureLegacyTree(owner).pipe(
    Effect.provideService(ProcessGroupController, processes), Effect.provideService(LegacyProcessTable, table))
  const verifyExecutable = (owner: LegacyOwner) => Effect.gen(function* () {
    const observed = process.platform === "linux"
      ? yield* Effect.tryPromise({ try: async () => ({ uid: (await stat(`/proc/${owner.pid}`)).uid, executable: await readlink(`/proc/${owner.pid}/exe`) }), catch: error => new PreviousInstallationFailed({ message: `Cannot inspect previous executable: ${String(error)}` }) })
      : yield* commands.run("/bin/ps", ["-p", String(owner.pid), "-o", "uid=,comm="]).pipe(Effect.flatMap(result => {
        const row = result.stdout.trim().match(/^(\d+)\s+(.+)$/)
        return result.code === 0 && row ? Effect.succeed({ uid: Number(row[1]), executable: row[2]! })
          : Effect.fail(new PreviousInstallationFailed({ message: "Cannot inspect the previous service process." }))
      }))
    const executable = observed.executable
    const canonical = yield* Effect.tryPromise({ try: async () => ({ executable: await realpath(executable), data: await realpath(options.dataDirectory), home: await realpath(options.home) }), catch: error => new PreviousInstallationFailed({ message: `Cannot verify previous executable: ${String(error)}` }) })
    const legacyBundle = join(canonical.home, "Applications/Magnitude.app/Contents/MacOS/magnitude-service")
    if (observed.uid !== process.getuid!() || basename(executable) !== "magnitude-service" ||
      !(inside(join(canonical.data, "releases/acn"), canonical.executable) || canonical.executable === legacyBundle)) {
      return yield* new PreviousInstallationFailed({ message: "The previous service does not match a Magnitude installation owned by this user." })
    }
  })
  return PreviousInstallation.of({
    inspect: Effect.gen(function* () {
      const registration = yield* startup.inspect
      // Broken obsolete records are not a startup gate. A live conflict remains protected by bind preflight.
      const record = yield* read.pipe(Effect.catchAll(error => Effect.logWarning("Cannot read old Magnitude ownership", error.message).pipe(Effect.as(Option.none<LegacyOwner>()))))
      const owner = record
      const identity = Option.isSome(owner) ? yield* processes.inspect(owner.value.pid) : Option.none()
      const live = Option.isSome(owner) && Option.exists(identity, value => value.processStartIdentity === owner.value.processStartIdentity)
      const registeredPid = Option.flatMap(registration, value => value.runningPid)
      if (Option.isSome(registeredPid) && (!live || !Option.exists(owner, value => value.pid === registeredPid.value))) {
        return yield* new PreviousInstallationFailed({ message: "The previous startup service is still initializing or its identity could not be verified." })
      }
      if (!live || Option.isNone(owner)) return Option.isNone(registration) ? Option.none()
        : Option.some({ _tag: "Unix" as const, startup: registration, tree: Option.none() })
      yield* verifyExecutable(owner.value)
      const tree = yield* capture(owner.value)
      return Option.some({ _tag: "Unix" as const, startup: registration, tree: Option.some(tree) })
    }).pipe(Effect.mapError(failed)),
    retire: (plan, checkpoint) => Effect.gen(function* () {
      // Resume with all saved groups, including children orphaned by an interrupted shutdown.
      let retirement = plan
      // A live service may have started another inference group since preparation.
      if (Option.isSome(plan.tree)) {
        const expected = plan.tree.value
        const root = yield* processes.inspect(expected.owner.pid)
        if (Option.isSome(root)) {
          if (root.value.processStartIdentity !== expected.owner.processStartIdentity) return yield* new PreviousInstallationFailed({ message: "Previous Magnitude process identity changed; no process was stopped." })
          yield* verifyExecutable(expected.owner)
          const current = yield* capture(expected.owner)
          if (!Schema.equivalence(Schema.Array(ProcessGroupSchema))(current.groups, expected.groups)) {
            const groups = [...expected.groups]
            for (const group of current.groups) {
              const saved = groups.find(value => value.leader.pid === group.leader.pid)
              if (saved && saved.leader.processStartIdentity !== group.leader.processStartIdentity) {
                return yield* new PreviousInstallationFailed({ message: "Previous inference process identity changed during upgrade." })
              }
              if (!saved) groups.push(group)
            }
            const tree = { ...expected, groups: [groups[0]!, ...groups.slice(1)] as const }
            retirement = { ...plan, tree: Option.some(tree) }
            yield* checkpoint(retirement)
          }
        }
      }
      const stopProcesses = Effect.gen(function* () {
        if (Option.isSome(retirement.tree)) {
          const tree = retirement.tree.value
          const groups = [...tree.groups].sort((a, b) => Number(b.leader.pid === tree.owner.pid) - Number(a.leader.pid === tree.owner.pid))
          for (const group of groups) {
            const result = yield* processes.stop(group)
            if (result._tag !== "ProcessGroupStopped") return yield* new PreviousInstallationFailed({ message: "An old process identity changed during shutdown." })
          }
          for (const group of groups) if ((yield* processes.observe(group))._tag !== "ProcessGroupAbsent") {
            return yield* new PreviousInstallationFailed({ message: "Previous Magnitude inference processes have not exited." })
          }
        }
      }).pipe(Effect.mapError(error => new LegacyStartupFailed({ message: error.message })))
      if (Option.isSome(retirement.startup)) yield* startup.unregister(retirement.startup.value, stopProcesses)
      else yield* stopProcesses
    }).pipe(Effect.mapError(failed)),
  })
})
