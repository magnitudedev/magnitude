import { BunContext } from "@effect/platform-bun"
import { DateTime, Deferred, Effect, Fiber, Layer, Option, Redacted, Schema, TestClock, TestContext } from "effect"
import { existsSync, readFileSync, statSync } from "node:fs"
import { expect, test } from "vitest"
import { LeaseId, RunId } from "../src/domain"
import { AzureMachine, MachineTags } from "../src/machines"
import { WorkerLaunch } from "../src/outward-runner"
import { azureBootstrap } from "../src/providers/azure-bootstrap"
import { ProcessExecutor, ProcessExecutorLive, checkedCommand, type CommandSpec } from "../src/process"

const config = { executable: "az", subscription: "5304c4b3-d605-4193-b0cb-766c065acfa6", resourceGroup: "magnitude-ci", adminUsername: "labworker" }
const name = "ml-123456789abc"
const id = `/subscriptions/${config.subscription}/resourceGroups/magnitude-ci/providers/Microsoft.Compute/virtualMachines/${name}`
const machine = () => AzureMachine.make({ provider: "azure", id, name, tags: MachineTags.make({ schemaVersion: 1,
  runId: RunId.make(`run-${crypto.randomUUID()}`), leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), expiresAt: DateTime.unsafeMake(Date.now() + 60_000) }) })
const launch = () => WorkerLaunch.make({ executable: "/bin/sh", args: ["-c", "printf '%s\\n' \"$1\"; umask", "lab-worker", "quote' space $HOME $(exit 42) `exit 43`"],
  root: "/tmp/lab attempt'", origin: "https://lab.example.com", token: Redacted.make("fixture-worker-secret"), deadline: DateTime.unsafeMake(Date.now() + 60_000) })
const output = (value: unknown) => ({ stdout: JSON.stringify(value), stderr: "", exitCode: 0 })

for (const mode of ["success", "foreign-scope", "foreign-lease", "windows", "windows-success", "expired", "delivery-error"] as const) test(`Azure worker delivery: ${mode}`, async () => {
  const vm = machine(), invocation = mode === "windows-success" ? WorkerLaunch.make({ ...launch(),
    executable: "C:\\Lab Runtime\\bun.exe", args: ["C:\\Lab Runtime\\worker.ts", "quote\" space $HOME `exit`", "", "tail\\"], root: "C:\\Users\\labworker\\Lab runs" }) : launch()
  const requests: CommandSpec[] = [], files: string[] = []
  let script = ""
  const executor = Layer.succeed(ProcessExecutor, { run: (spec: CommandSpec) => Effect.sync(() => {
    requests.push(spec)
    expect(spec.args.join(" ")).not.toContain(Redacted.value(invocation.token))
    expect(spec.args[spec.args.indexOf("--subscription") + 1]).toBe(config.subscription)
    if (spec.args.includes("GET")) return output({ id, name, location: "westus2", tags: {
      "lab-owner": "magnitude-testing-lab-v1", "lab-machine": name,
      "lab-lease": Schema.encodeSync(Schema.parseJson(MachineTags))(mode === "foreign-lease" ? machine().tags : vm.tags),
    }, properties: { provisioningState: "Succeeded", storageProfile: { osDisk: { osType: mode.startsWith("windows") ? "Windows" : "Linux" } } } })
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
    const bootstrap = yield* azureBootstrap(config)
    yield* bootstrap.start(mode === "foreign-scope" ? { ...vm, id: id.replace("magnitude-ci", "another-group") } : vm,
      mode === "expired" ? { ...invocation, deadline: DateTime.unsafeMake(0) } : invocation)
  }).pipe(Effect.either, Effect.provide(Layer.merge(BunContext.layer, executor))))
  expect(result._tag).toBe(mode === "success" || mode === "windows-success" ? "Right" : "Left")
  if (result._tag === "Left") expect(result.left.message).not.toContain(Redacted.value(invocation.token))
  for (const file of files) expect(existsSync(file)).toBe(false)
  expect(requests.length).toBe(mode === "foreign-scope" || mode === "expired" ? 0 : mode === "success" || mode === "windows-success" || mode === "delivery-error" ? 2 : 1)
  if (mode === "windows-success") {
    const encoded = script.match(/FromBase64String\('([A-Za-z0-9+/=]+)'\)/)![1]!
    expect(JSON.parse(Buffer.from(encoded, "base64").toString())).toMatchObject({ user: "labworker", executable: invocation.executable,
      root: invocation.root, args: invocation.args, origin: invocation.origin })
    expect(script).toContain("-LogonType Interactive")
    expect(script).toContain(".SessionId -eq 0")
    expect(script).toContain("Unregister-ScheduledTask")
    expect(script).toContain("$info.LastRunTime -ne $initialRun")
    expect(script).not.toContain("$started.AddSeconds(-2)")
  }
  if (mode === "success") {
    // Execute the exact generated shell to catch interpolation bugs, rather than comparing quoting strings.
    const userSwitch = "/usr/bin/sudo -n -H --preserve-env=LAB_WORKER_TOKEN,LAB_WORKER_ROOT,LAB_URL -u 'labworker' -- "
    expect(script).toContain(userSwitch)
    // The user switch is qualified on Azure; execute the remaining exact shell locally for literal handling.
    const executed = await Effect.runPromise(checkedCommand("/bin/sh", ["-c", script.replace(userSwitch, "")], {
      inheritEnv: false, env: { LAB_WORKER_TOKEN: Redacted.value(invocation.token) },
    }).pipe(Effect.provide(ProcessExecutorLive)))
    expect(executed.stdout).toBe(`${invocation.args[3]}\n0022\n`)
  }
})

for (const state of ["Pending", "Running", "Succeeded", "Failed", "TimedOut", "Canceled", "missing-view", "foreign-command"] as const) test(`Azure native execution observation: ${state}`, () => Effect.runPromise(Effect.gen(function* () {
  const vm = machine()
  const executor = Layer.succeed(ProcessExecutor, { run: spec => Effect.gen(function* () {
    expect(spec.args).toContain("GET")
    expect(spec.args[spec.args.indexOf("--url") + 1]).toBe(`https://management.azure.com${id}/runCommands/lab-worker?api-version=2024-11-01&$expand=instanceView`)
    return { exitCode: 0, stderr: "", stdout: yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ id: `${id}/runCommands/${state === "foreign-command" ? "another-command" : "lab-worker"}`, properties: {
      provisioningState: "Succeeded", ...(state === "missing-view" ? {} : { instanceView: { executionState: state === "foreign-command" ? "Succeeded" : state,
        exitCode: state === "Failed" ? 17 : 0, output: "must not expose provider output or protected values", error: "private diagnostic fixture" } }),
    } }) }
  }).pipe(Effect.orDie) })
  const observed = yield* azureBootstrap(config).pipe(Effect.flatMap(bootstrap => bootstrap.poll(vm)), Effect.provide(executor), Effect.either)
  if (state === "foreign-command") expect(observed._tag).toBe("Left")
  else if (observed._tag === "Right") {
    expect(observed.right._tag).toBe(["Pending", "Running", "missing-view"].includes(state) ? "None" : "Some")
    if (observed.right._tag === "Some") expect(observed.right.value).toEqual({ state, code: Option.some(state === "Failed" ? 17 : 0), output: Option.some(Redacted.make("must not expose provider output or protected values\nprivate diagnostic fixture")) })
  } else expect.fail("Execution observation unexpectedly failed")
}).pipe(Effect.provide(BunContext.layer))))

for (const mode of ["settles", "foreign-after-wait", "failed", "unknown"] as const) test(`Azure bootstrap settles preparation without leaking authority: ${mode}`, async () => {
  const vm = machine(), invocation = launch()
  let reads = 0, deliveries = 0
  const executor = Layer.succeed(ProcessExecutor, { run: (spec: CommandSpec) => Effect.sync(() => {
    if (spec.args.includes("PUT")) { deliveries++; return output({}) }
    reads++
    const state = mode === "failed" ? "Failed" : mode === "unknown" ? "Unexpected" : reads === 1 ? "Updating" : "Succeeded"
    return output({ id, name, location: "westus2", tags: {
      "lab-owner": "magnitude-testing-lab-v1", "lab-machine": name,
      "lab-lease": Schema.encodeSync(Schema.parseJson(MachineTags))(mode === "foreign-after-wait" && reads > 1 ? machine().tags : vm.tags),
    }, properties: { provisioningState: state, storageProfile: { osDisk: { osType: "Linux" } } } })
  }) })
  const result = await Effect.runPromise(azureBootstrap(config).pipe(Effect.flatMap(bootstrap => bootstrap.start(vm, invocation)),
    Effect.either, Effect.provide(Layer.merge(BunContext.layer, executor))))
  expect(result._tag).toBe(mode === "settles" ? "Right" : "Left")
  expect(deliveries).toBe(mode === "settles" ? 1 : 0)
  expect(reads).toBe(mode === "settles" || mode === "foreign-after-wait" ? 2 : 1)
})


test("Azure provisioning wait stops at the allocation deadline before delivering authority", () => Effect.runPromise(Effect.gen(function* () {
  const base = machine(), vm = { ...base, tags: { ...base.tags, expiresAt: DateTime.unsafeMake(60_000) } }
  const invocation = { ...launch(), deadline: DateTime.unsafeMake(600_000) }
  const observed = yield* Deferred.make<void>()
  let deliveries = 0
  const executor = Layer.succeed(ProcessExecutor, { run: (spec: CommandSpec) => Effect.gen(function* () {
    if (spec.args.includes("PUT")) { deliveries++; return output({}) }
    yield* Deferred.succeed(observed, undefined)
    return output({ id, name, location: "westus2", tags: {
      "lab-owner": "magnitude-testing-lab-v1", "lab-machine": name,
      "lab-lease": yield* Schema.encode(Schema.parseJson(MachineTags))(vm.tags).pipe(Effect.orDie),
    }, properties: { provisioningState: "Updating", storageProfile: { osDisk: { osType: "Linux" } } } })
  }) })
  const fiber = yield* azureBootstrap(config).pipe(Effect.flatMap(bootstrap => bootstrap.start(vm, invocation)),
    Effect.either, Effect.provide(executor), Effect.fork)
  yield* Deferred.await(observed)
  yield* TestClock.adjust("60 seconds")
  const result = yield* Fiber.join(fiber)
  expect(result._tag).toBe("Left")
  if (result._tag === "Left") expect(result.left.message).toContain("bootstrap deadline")
  expect(deliveries).toBe(0)
}).pipe(Effect.provide(Layer.merge(BunContext.layer, TestContext.TestContext)))))
