import { DateTime, Effect, Option, Schema } from "effect"
import { InfrastructureFailure } from "../domain"
import { ProcessExecutor } from "../process"
import { AzureMachine } from "../machines"
import { azureRuntimeDownload, prepareAzureInitialization } from "./azure-initialization"
import { WindowsRuntimePreparation, windowsDesktopReadiness } from "./windows-initialization"

const api = "2024-11-01"
const fail = (message: string) => new InfrastructureFailure({ operation: "azure-windows-preparation", message })
type Prepared = Extract<Effect.Effect.Success<ReturnType<typeof prepareAzureInitialization>>, { kind: "windows" }>
export interface WindowsPreparationOperations {
  readonly rest: (method: string, id: string, version: string, body?: unknown) => Effect.Effect<{ readonly stdout: string }, InfrastructureFailure>
  readonly restart: Effect.Effect<unknown, InfrastructureFailure>
  readonly waitProvisioned: Effect.Effect<void, InfrastructureFailure>
}
const Tags = Schema.Record({ key: Schema.String, value: Schema.String })
const Observation = Schema.Struct({ tags: Tags, properties: Schema.Struct({ instanceView: Schema.optionalWith(Schema.Struct({
  executionState: Schema.String, exitCode: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }),
}), { as: "Option", exact: true }) }) })
const Inventory = Schema.Struct({ value: Schema.Array(Schema.Struct({ id: Schema.String })) })
const stages = ["lab-tools", "lab-runtime", "lab-desktop", "lab-ready"] as const

/** Reconcile provider stage identities; never rerun a failed or ambiguously accepted native installer. */
export const prepareWindowsMachine = (machine: typeof AzureMachine.Type, prepared: Prepared,
  scope: { readonly executable: string; readonly subscription: string; readonly location: string }, operations: WindowsPreparationOperations,
) => Effect.gen(function* () {
  const { rest } = operations
  const remaining = () => DateTime.toEpochMillis(machine.tags.expiresAt) - Date.now()
  const readTags = () => rest("GET", machine.id, api).pipe(Effect.flatMap(reply => Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ tags: Tags })))(reply.stdout)),
    Effect.flatMap(vm => vm.tags["lab-initialization"] === prepared.identity ? Effect.succeed(vm.tags) : Effect.fail(fail("Windows VM preparation identity differs from its lease"))))
  yield* readTags()
  const inventory = () => rest("GET", `${machine.id}/runCommands`, api).pipe(Effect.flatMap(reply => Schema.decodeUnknown(Schema.parseJson(Inventory))(reply.stdout)))
  const stage = (name: typeof stages[number], script: string, seconds: number,
    parameters: Effect.Effect<ReadonlyArray<{ readonly name: string; readonly value: string }>, InfrastructureFailure, ProcessExecutor> = Effect.succeed([]),
  ) => Effect.gen(function* () {
    if (remaining() <= 0) return yield* fail("Windows preparation lease expired")
    // A fresh read-only readiness command prevents Azure from returning the previous execution's instanceView.
    const commandName = name === "lab-ready" ? `${name}-${crypto.randomUUID()}` : name
    const id = `${machine.id}/runCommands/${commandName}`
    const exists = (yield* inventory()).value.some(item => item.id.toLowerCase() === id.toLowerCase())
    if (!exists) {
      const protectedParameters = yield* parameters
      const submitted = yield* rest("PUT", id, api, { location: scope.location, tags: { "lab-initialization": prepared.identity }, properties: {
        source: { script }, protectedParameters, asyncExecution: true,
        timeoutInSeconds: Math.max(1, Math.min(seconds, Math.floor(remaining() / 1000))),
      } }).pipe(Effect.either)
      // A lost response can still have created the command. Observe it, never submit it twice.
      if (submitted._tag === "Left" && !(yield* inventory()).value.some(item => item.id.toLowerCase() === id.toLowerCase())) return yield* submitted.left
    }
    for (;;) {
      if (remaining() <= 0) return yield* fail(`Windows ${name} exceeded its lease`)
      const result = yield* rest("GET", id, `${api}&$expand=instanceView`).pipe(Effect.flatMap(reply => Schema.decodeUnknown(Schema.parseJson(Observation))(reply.stdout)))
      if (result.tags["lab-initialization"] !== prepared.identity) return yield* fail(`Windows ${name} belongs to a different preparation recipe`)
      if (Option.isSome(result.properties.instanceView)) {
        const view = result.properties.instanceView.value
        if (view.executionState === "Succeeded") {
          if (Option.isNone(view.exitCode) || view.exitCode.value !== 0) return yield* fail(`Windows ${name} has no successful native exit`)
          return
        }
        if (["Failed", "Canceled", "TimedOut"].includes(view.executionState)) return yield* fail(`Windows ${name} ended in ${view.executionState}`)
      }
      yield* Effect.sleep("5 seconds")
    }
  }).pipe(Effect.timeoutFail({ duration: Math.max(1, Math.min((seconds + 180) * 1000, remaining())), onTimeout: () => fail(`Windows ${name} exceeded its observation deadline; allocation remains owned for cleanup`) }))
  yield* stage("lab-tools", prepared.toolsScript, 2400)
  const configuration = Effect.gen(function* () {
    const runtime = yield* azureRuntimeDownload(prepared.recipe.runtime, scope)
    const encoded = yield* Schema.encode(Schema.parseJson(WindowsRuntimePreparation))({ distribution: prepared.recipe.distribution,
      adminUsername: prepared.recipe.adminUsername, architecture: prepared.recipe.architecture, runtime })
    return [{ name: "LAB_INITIALIZATION", value: Buffer.from(encoded).toString("base64") }]
  }).pipe(Effect.mapError(() => fail("Cannot prepare the protected Windows runtime configuration")))
  yield* stage("lab-runtime", prepared.runtimeScript, 2400, configuration)
  yield* stage("lab-desktop", prepared.desktopScript, 180)
  const tags = yield* readTags()
  if (!tags["lab-desktop-restart"]) {
    // Persist intent before the restart. Ambiguity may fail readiness, but must never reboot an already-ready session.
    yield* rest("PATCH", machine.id, api, { tags: { ...tags, "lab-desktop-restart": prepared.identity } })
    yield* operations.restart
  } else if (tags["lab-desktop-restart"] !== prepared.identity) return yield* fail("Windows restart belongs to another preparation recipe")
  yield* operations.waitProvisioned
  yield* stage("lab-ready", yield* windowsDesktopReadiness(prepared.recipe.runtime.sha256, prepared.recipe.adminUsername), 180)
}).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : fail("Invalid Windows preparation observation")),
  Effect.timeoutFail({ duration: Math.max(1, DateTime.toEpochMillis(machine.tags.expiresAt) - Date.now()),
    onTimeout: () => fail("Windows preparation exceeded its lease; allocation remains owned for cleanup") }))

/** Read only execution diagnostics, never protected parameters or command source containing configuration. */
export const windowsPreparationDiagnostics = (machine: typeof AzureMachine.Type, rest: WindowsPreparationOperations["rest"]) => Effect.gen(function* () {
  const inventory = yield* rest("GET", `${machine.id}/runCommands`, api).pipe(Effect.flatMap(reply => Schema.decodeUnknown(Schema.parseJson(Inventory))(reply.stdout)))
  const Diagnostic = Schema.Struct({ properties: Schema.Struct({ instanceView: Schema.optionalWith(Schema.Struct({
    executionState: Schema.String, output: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    error: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }), { as: "Option", exact: true }) }) })
  const output: string[] = []
  const prefix = `${machine.id}/runCommands/`
  for (const item of inventory.value) {
    if (!item.id.toLowerCase().startsWith(prefix.toLowerCase())) continue
    const name = item.id.slice(prefix.length)
    if (!stages.includes(name as typeof stages[number]) && !/^lab-ready-[a-f0-9-]{36}$/.test(name)) continue
    const id = item.id
    const observed = yield* rest("GET", id, `${api}&$expand=instanceView`).pipe(Effect.flatMap(reply => Schema.decodeUnknown(Schema.parseJson(Diagnostic))(reply.stdout)))
    if (Option.isSome(observed.properties.instanceView)) {
      const view = observed.properties.instanceView.value
      output.push(`${name}: ${view.executionState}\n${Option.getOrElse(view.output, () => "").slice(-8000)}\n${Option.getOrElse(view.error, () => "").slice(-8000)}`)
    }
  }
  return output.join("\n")
})
