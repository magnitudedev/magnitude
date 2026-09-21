import { FileSystem } from "@effect/platform"
import { Clock, DateTime, Effect, Option, Redacted, Schedule, Schema } from "effect"
import { join, posix } from "node:path"
import { InfrastructureFailure } from "../domain"
import { type Machine, MachineTags } from "../machines"
import { type WorkerBootstrap, WorkerExit } from "../outward-runner"
import { checkedCommand, ProcessExecutor } from "../process"
import { AzureConfig } from "./azure"
import { windowsInteractiveScript } from "./windows-interactive"

export const AzureBootstrapConfig = AzureConfig.pick("executable", "subscription", "resourceGroup", "adminUsername")
const Observation = Schema.Struct({ id: Schema.String, name: Schema.String, location: Schema.NonEmptyString,
  tags: Schema.Record({ key: Schema.String, value: Schema.String }),
  properties: Schema.Struct({ provisioningState: Schema.String,
    storageProfile: Schema.Struct({ osDisk: Schema.Struct({ osType: Schema.Literal("Linux", "Windows") }) }) }) })
const fail = (message: string) => new InfrastructureFailure({ operation: "azure-bootstrap", message })
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`

/** Managed Run Command delivers authority; the guest reports results through the worker API.
 * Images must already contain their configured runtime and desktop dependencies.
 * Windows additionally requires an active session for the admitted desktop user.
 */
export const azureBootstrap = (config: typeof AzureBootstrapConfig.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* ProcessExecutor
  const group = `/subscriptions/${config.subscription}/resourceGroups/${config.resourceGroup}`
  const request = (method: "GET" | "PUT", id: string, body?: unknown, expand = false) => Effect.scoped(Effect.gen(function* () {
    const args = ["rest", "--subscription", config.subscription, "--method", method, "--url",
      `https://management.azure.com${id}?api-version=2024-11-01${expand ? "&$expand=instanceView" : ""}`, "--only-show-errors", "--output", "json"]
    if (body !== undefined) {
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-lab-bootstrap-" })
      const file = join(directory, "request.json")
      yield* fs.writeFileString(file, yield* Schema.encode(Schema.parseJson(Schema.Unknown))(body), { mode: 0o600 })
      args.push("--body", `@${file}`)
    }
    return yield* checkedCommand(config.executable, args, { timeoutMs: method === "GET" ? 30_000 : 180_000 }).pipe(Effect.provideService(ProcessExecutor, executor))
  })).pipe(Effect.mapError(() => fail(`Azure ${method} worker delivery failed; inspect the provider activity log`)))
  const verifyScope = (machine: Machine) => Effect.gen(function* () {
    if (machine.provider !== "azure" || !/^ml-[a-f0-9]{12}$/.test(machine.name) ||
      machine.id.toLowerCase() !== `${group}/providers/Microsoft.Compute/virtualMachines/${machine.name}`.toLowerCase()) return yield* fail("Worker is outside the configured Azure scope")
    return machine
  })
  const Execution = Schema.Struct({ id: Schema.String, properties: Schema.Struct({
    provisioningState: Schema.String,
    instanceView: Schema.optionalWith(Schema.Struct({ executionState: Schema.Literal("Unknown", "Pending", "Running", "Succeeded", "Failed", "TimedOut", "Canceled"),
      output: Schema.optionalWith(Schema.NullOr(Schema.String), { as: "Option", exact: true }),
      error: Schema.optionalWith(Schema.NullOr(Schema.String), { as: "Option", exact: true }),
      executionMessage: Schema.optionalWith(Schema.NullOr(Schema.String), { as: "Option", exact: true }),
      exitCode: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }) }), { as: "Option", exact: true }),
  }) })
  return {
    poll: machine => Effect.gen(function* () {
      const vm = yield* verifyScope(machine)
      const id = `${vm.id}/runCommands/lab-worker`
      const observed = yield* request("GET", id, undefined, true).pipe(
        Effect.retry({ times: 2, schedule: Schedule.spaced("1 second") }),
        Effect.flatMap(result => Schema.decodeUnknown(Schema.parseJson(Execution))(result.stdout)),
        Effect.mapError(() => fail("Cannot read native worker execution state")))
      if (observed.id.toLowerCase() !== id.toLowerCase()) return yield* fail("Execution state belongs to another worker")
      const state = observed.properties.instanceView
      if (Option.isSome(state) && ["Succeeded", "Failed", "TimedOut", "Canceled"].includes(state.value.executionState)) {
        const output = [state.value.output, state.value.error, state.value.executionMessage].flatMap(value => Option.toArray(value).filter((text): text is string => typeof text === "string")).join("\n").slice(-32 * 1024)
        return Option.some(yield* Schema.decodeUnknown(WorkerExit)({ state: state.value.executionState,
          ...(output ? { output } : {}), ...Option.match(state.value.exitCode, { onNone: () => ({}), onSome: code => ({ code }) }) }).pipe(Effect.orDie))
      }
      if (["Failed", "Canceled"].includes(observed.properties.provisioningState)) return Option.some(WorkerExit.make({ state: "Failed", code: Option.none() }))
      return Option.none()
    }),
    start: (machine, launch) => Effect.gen(function* () {
      const owned = yield* verifyScope(machine)
      const deadline = Math.min(DateTime.toEpochMillis(launch.deadline), DateTime.toEpochMillis(owned.tags.expiresAt))
      const now = yield* Clock.currentTimeMillis
      if (deadline <= now) return yield* fail("Cannot start an expired worker")
      // VM tags and desktop preparation can leave Azure briefly Updating after
      // guest readiness. Revalidate ownership on every read before issuing authority.
      const vm = yield* Effect.gen(function* () {
        for (;;) {
          const observed = yield* request("GET", owned.id).pipe(Effect.flatMap(result => Schema.decodeUnknown(Schema.parseJson(Observation))(result.stdout)),
            Effect.mapError(() => fail("Cannot verify Azure worker identity")))
          const tags = yield* Schema.decodeUnknown(Schema.parseJson(MachineTags))(observed.tags["lab-lease"]).pipe(Effect.mapError(() => fail("Worker has invalid lease metadata")))
          if (observed.id.toLowerCase() !== owned.id.toLowerCase() || observed.name !== owned.name ||
            observed.tags["lab-owner"] !== "magnitude-testing-lab-v1" || observed.tags["lab-machine"] !== owned.name ||
            !Schema.equivalence(MachineTags)(tags, owned.tags)) return yield* fail("Worker ownership changed before credential delivery")
          const state = observed.properties.provisioningState
          if (state === "Succeeded") return observed
          if (state !== "Updating" && state !== "Creating") return yield* fail(`Worker cannot start from Azure provisioning state ${state}`)
          yield* Effect.sleep("1 second")
        }
      }).pipe(Effect.timeoutFail({ duration: Math.min(120_000, deadline - now),
        onTimeout: () => fail("Worker provisioning did not settle before its bootstrap deadline") }))
      const windows = vm.properties.storageProfile.osDisk.osType === "Windows"
      if (!windows && (!posix.isAbsolute(launch.executable) || !posix.isAbsolute(launch.root) ||
        [launch.executable, launch.root, launch.origin, ...launch.args].some(value => value.includes("\0")))) return yield* fail("Linux worker paths must be absolute and launch arguments must not contain NUL")
      if (deadline <= (yield* Clock.currentTimeMillis)) return yield* fail("Worker expired while verifying its identity")
      // Azure's runAsUser uses sudo without preserving named protected parameters.
      // Receive them as root, then retain only the lab variables through an explicit user switch.
      // The credential remains an environment value, never a command argument or script literal.
      // Package builders require normal directory permissions. Invocation credentials and
      // workspaces use explicit 0600/0700 modes instead of imposing 077 on their children.
      const script = windows ? yield* windowsInteractiveScript({ user: config.adminUsername, executable: launch.executable, args: launch.args,
        root: launch.root, origin: launch.origin, timeoutSeconds: Math.max(1, Math.ceil((deadline - (yield* Clock.currentTimeMillis)) / 1000)) }) : ["#!/bin/sh", "set -eu", "umask 022", ': "${LAB_WORKER_TOKEN:?Missing worker credential}"',
        `export LAB_WORKER_ROOT=${quote(launch.root)}`, `export LAB_URL=${quote(launch.origin)}`,
        "cd /",
        `exec /usr/bin/sudo -n -H --preserve-env=LAB_WORKER_TOKEN,LAB_WORKER_ROOT,LAB_URL -u ${quote(config.adminUsername)} -- ${[launch.executable, ...launch.args].map(quote).join(" ")}`].join("\n")
      yield* request("PUT", `${owned.id}/runCommands/lab-worker`, { location: vm.location, properties: {
        source: { script }, asyncExecution: true,
        timeoutInSeconds: Math.max(1, Math.ceil((deadline - (yield* Clock.currentTimeMillis)) / 1000)),
        protectedParameters: [{ name: "LAB_WORKER_TOKEN", value: Redacted.value(launch.token) }],
      } })
    }),
  } satisfies WorkerBootstrap
})
