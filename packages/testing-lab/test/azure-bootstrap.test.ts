import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Redacted, Schema } from "effect"
import { existsSync, readFileSync, statSync } from "node:fs"
import { expect, test } from "vitest"
import { LeaseId, RunId } from "../src/domain"
import { AzureMachine, MachineTags } from "../src/machines"
import { WorkerLaunch } from "../src/outward-runner"
import { azureLinuxBootstrap } from "../src/providers/azure-bootstrap"
import { ProcessExecutor, ProcessExecutorLive, checkedCommand, type CommandSpec } from "../src/process"

const config = { executable: "az", subscription: "5304c4b3-d605-4193-b0cb-766c065acfa6", resourceGroup: "magnitude-ci", adminUsername: "labworker" }
const name = "ml-123456789abc"
const id = `/subscriptions/${config.subscription}/resourceGroups/magnitude-ci/providers/Microsoft.Compute/virtualMachines/${name}`
const machine = () => AzureMachine.make({ provider: "azure", id, name, tags: MachineTags.make({ schemaVersion: 1,
  runId: RunId.make(`run-${crypto.randomUUID()}`), leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), expiresAt: DateTime.unsafeMake(Date.now() + 60_000) }) })
const launch = () => WorkerLaunch.make({ executable: "/usr/bin/printf", args: ["%s\\n", "quote' space $HOME $(exit 42) `exit 43`"],
  root: "/tmp/lab attempt'", origin: "https://lab.example.com", token: Redacted.make("fixture-worker-secret"), deadline: DateTime.unsafeMake(Date.now() + 60_000) })
const output = (value: unknown) => ({ stdout: JSON.stringify(value), stderr: "", exitCode: 0 })

for (const mode of ["success", "foreign-scope", "foreign-lease", "windows", "expired", "delivery-error"] as const) test(`Azure worker delivery: ${mode}`, async () => {
  const vm = machine(), invocation = launch()
  const requests: CommandSpec[] = [], files: string[] = []
  let script = ""
  const executor = Layer.succeed(ProcessExecutor, { run: (spec: CommandSpec) => Effect.sync(() => {
    requests.push(spec)
    expect(spec.args.join(" ")).not.toContain(Redacted.value(invocation.token))
    expect(spec.args[spec.args.indexOf("--subscription") + 1]).toBe(config.subscription)
    if (spec.args.includes("GET")) return output({ id, name, location: "westus2", tags: {
      "lab-owner": "magnitude-testing-lab-v1", "lab-machine": name,
      "lab-lease": Schema.encodeSync(Schema.parseJson(MachineTags))(mode === "foreign-lease" ? machine().tags : vm.tags),
    }, properties: { provisioningState: "Succeeded", storageProfile: { osDisk: { osType: mode === "windows" ? "Windows" : "Linux" } } } })
    const file = spec.args[spec.args.indexOf("--body") + 1]!.slice(1)
    files.push(file)
    expect(statSync(file).mode & 0o777).toBe(0o600)
    const body = JSON.parse(readFileSync(file, "utf8"))
    expect(body.properties.protectedParameters).toEqual([{ name: "LAB_WORKER_TOKEN", value: Redacted.value(invocation.token) }])
    expect(body.properties.runAsUser).toBeUndefined()
    expect(body.properties.asyncExecution).toBe(true)
    expect(body.properties.timeoutInSeconds).toBeGreaterThan(0)
    script = body.properties.source.script
    expect(script).not.toContain(Redacted.value(invocation.token))
    return mode === "delivery-error" ? { stdout: "", stderr: Redacted.value(invocation.token), exitCode: 1 } : output({})
  }) })
  const result = await Effect.runPromise(Effect.gen(function* () {
    const bootstrap = yield* azureLinuxBootstrap(config)
    yield* bootstrap.start(mode === "foreign-scope" ? { ...vm, id: id.replace("magnitude-ci", "another-group") } : vm,
      mode === "expired" ? { ...invocation, deadline: DateTime.unsafeMake(0) } : invocation)
  }).pipe(Effect.either, Effect.provide(Layer.merge(BunContext.layer, executor))))
  expect(result._tag).toBe(mode === "success" ? "Right" : "Left")
  if (result._tag === "Left") expect(result.left.message).not.toContain(Redacted.value(invocation.token))
  for (const file of files) expect(existsSync(file)).toBe(false)
  expect(requests.length).toBe(mode === "foreign-scope" || mode === "expired" ? 0 : mode === "success" || mode === "delivery-error" ? 2 : 1)
  if (mode === "success") {
    // Execute the exact generated shell to catch interpolation bugs, rather than comparing quoting strings.
    const userSwitch = "/usr/bin/sudo -n -H --preserve-env=LAB_WORKER_TOKEN,LAB_WORKER_ROOT,LAB_URL -u 'labworker' -- "
    expect(script).toContain(userSwitch)
    // The user switch is qualified on Azure; execute the remaining exact shell locally for literal handling.
    const executed = await Effect.runPromise(checkedCommand("/bin/sh", ["-c", script.replace(userSwitch, "")], {
      inheritEnv: false, env: { LAB_WORKER_TOKEN: Redacted.value(invocation.token) },
    }).pipe(Effect.provide(ProcessExecutorLive)))
    expect(executed.stdout).toBe(`${invocation.args[1]}\n`)
  }
})
