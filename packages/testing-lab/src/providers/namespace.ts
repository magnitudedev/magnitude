import { DateTime, Effect, Layer, Schema } from "effect"
import { InfrastructureFailure } from "../domain"
import { MachineAllocator, MachineTags, NamespaceMachine, WorkerTransport } from "../machines"
import { checkedCommand, command, ProcessExecutor } from "../process"

export const NamespaceImage = Schema.Struct({ version: Schema.String, selector: Schema.String,
  catalogCreatedAt: Schema.String, productVersion: Schema.String, buildVersion: Schema.String })
export type NamespaceImage = typeof NamespaceImage.Type
const Box = Schema.Struct({ id: Schema.String, name: Schema.String,
  documented_purpose: Schema.optionalWith(Schema.String, { as: "Option", exact: true }) })
const Image = Schema.Struct({ name: Schema.String, created_at: Schema.String })
const marker = "magnitude-lab/v1 "
const failed = (message: string) => new InfrastructureFailure({ operation: "namespace", message })
export const namespaceAllocator = <R = never>(executable: string, images: ReadonlyArray<NamespaceImage>,
  prepare?: (machine: typeof NamespaceMachine.Type, image: NamespaceImage) => Effect.Effect<void, InfrastructureFailure, R>) => Layer.effect(MachineAllocator, Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const preparationContext = yield* Effect.context<R>()
  const checked = (...args: Parameters<typeof checkedCommand>) => checkedCommand(...args).pipe(Effect.provideService(ProcessExecutor, executor))
  const list = () => checked(executable, ["list", "--output", "json"]).pipe(Effect.flatMap(output => Effect.gen(function* () {
    // devbox 0.0.189 prefixes its empty JSON inventory with this human-facing notice.
    const notice = "No devbox available yet. Try running `devbox create`."
    const lines = output.stdout.replaceAll("\r\n", "\n")
    const emptyNotice = lines.startsWith(`${notice}\n`)
    const boxes = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Array(Box)))(emptyNotice ? lines.slice(notice.length + 1) : lines)
    if (emptyNotice && boxes.length !== 0) return yield* failed("Namespace empty-inventory notice contradicts its machine list")
    return boxes
  })), Effect.mapError(() => failed("Cannot read Namespace inventory")))
  const tagged = (box: typeof Box.Type) => Effect.gen(function* () {
    if (box.documented_purpose._tag !== "Some" || !box.documented_purpose.value.startsWith(marker)) return yield* failed("Machine is not owned by the lab")
    const tags = yield* Schema.decodeUnknown(Schema.parseJson(MachineTags))(box.documented_purpose.value.slice(marker.length)).pipe(Effect.mapError(() => failed("Invalid Namespace ownership metadata")))
    return NamespaceMachine.make({ provider: "namespace", id: box.id, name: box.name, tags })
  })
  return {
    inventory: () => list().pipe(Effect.flatMap(boxes => Effect.forEach(boxes.filter(box => box.documented_purpose._tag === "Some" && box.documented_purpose.value.startsWith(marker)), tagged))),
    ensure: (lease, target) => Effect.gen(function* () {
      if (lease.provider !== "namespace" || target.os !== "macos" || target.arch !== "arm64") return yield* failed("Namespace allocator requires an Apple Silicon macOS target")
      if (DateTime.toEpochMillis(lease.expiresAt) <= Date.now()) return yield* failed("Cannot provision an expired lease")
      const image = images.find(i => i.version === target.version)
      if (!image) return yield* failed(`No qualified image lock for macOS ${target.version}`)
      const tags = MachineTags.make({ schemaVersion: 1, runId: lease.runId, leaseId: lease.leaseId, expiresAt: lease.expiresAt })
      let box = (yield* list()).find(b => b.name === lease.resourceName)
      if (!box) {
        const available = yield* checked(executable, ["image", "list", "--platform", "macos", "--output", "json"]).pipe(
          Effect.flatMap(output => Schema.decodeUnknown(Schema.parseJson(Schema.Array(Image)))(output.stdout)),
          Effect.mapError(() => failed("Cannot verify Namespace image catalog")))
        if (!available.some(i => i.name === image.selector && i.created_at === image.catalogCreatedAt)) return yield* failed("Namespace image catalog changed; requalify the image lock before allocation")
        const purpose = marker + (yield* Schema.encode(Schema.parseJson(MachineTags))(tags).pipe(Effect.orDie))
        const attempt = yield* checked(executable, ["create", "--name", lease.resourceName, "--platform", "macos", "--image", image.selector,
          "--size", "m", "--no_checkout", "--ephemeral", "--auto_stop_idle_timeout", "15m", "--access_mode", "private", "--purpose", purpose], { timeoutMs: 10 * 60_000 }).pipe(Effect.either)
        // Resolve a timed-out/ambiguous create by deterministic identity before any retry.
        box = (yield* list()).find(b => b.name === lease.resourceName)
        if (!box) return yield* failed(attempt._tag === "Left" ? "Namespace allocation failed and inventory contains no matching machine" : "Namespace create returned without a matching machine")
      }
      const machine = yield* tagged(box)
      if (!Schema.equivalence(MachineTags)(machine.tags, tags)) return yield* failed("Namespace resource name belongs to another lease")
      const probe = yield* checked(executable, ["exec", machine.name, "--", "/usr/bin/sw_vers"], { timeoutMs: 120_000 })
      if (!probe.stdout.includes(`ProductVersion:\t\t${image.productVersion}`) || !probe.stdout.includes(`BuildVersion:\t\t${image.buildVersion}`)) return yield* failed("Namespace guest OS does not match the qualified image lock")
      if (prepare) yield* prepare(machine, image).pipe(Effect.provide(preparationContext))
      return machine
    }),
    release: machine => Effect.gen(function* () {
      if (machine.provider !== "namespace") return yield* failed("Wrong allocator for this machine")
      const box = (yield* list()).find(b => b.id === machine.id)
      if (!box) return
      const owned = yield* tagged(box)
      if (!Schema.equivalence(MachineTags)(owned.tags, machine.tags)) return yield* failed("Refusing to delete a machine with different ownership metadata")
      yield* checked(executable, ["shutdown", box.name, "--force"], { timeoutMs: 180_000 })
      // Ephemeral devboxes can disappear during shutdown; inventory is authoritative.
      const remaining = (yield* list()).find(b => b.id === machine.id)
      if (!remaining) return
      const stillOwned = yield* tagged(remaining)
      if (!Schema.equivalence(MachineTags)(stillOwned.tags, machine.tags) || remaining.name !== box.name) return yield* failed("Namespace ownership changed during shutdown")
      yield* checked(executable, ["expire", box.name, "--force"], { timeoutMs: 180_000 })
      if ((yield* list()).some(b => b.id === machine.id)) return yield* failed("Namespace machine remains after expiration")
    }),
  } satisfies MachineAllocator
}))
export const namespaceTransport = (executable: string) => Layer.effect(WorkerTransport, Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const checked = (...args: Parameters<typeof checkedCommand>) => checkedCommand(...args).pipe(Effect.provideService(ProcessExecutor, executor))
  const run = (...args: Parameters<typeof command>) => command(...args).pipe(Effect.provideService(ProcessExecutor, executor))
  return {
  execute: (machine, program, args, timeoutMs) => machine.provider === "namespace"
    ? run(executable, ["exec", machine.name, "--", program, ...args], { timeoutMs }) : Effect.fail(failed("Wrong transport for this machine")),
  upload: (machine, local, remote) => machine.provider === "namespace"
    ? checked(executable, ["upload", machine.name, local, remote, "--mkdir"], { timeoutMs: 10 * 60_000 }).pipe(Effect.asVoid) : Effect.fail(failed("Wrong transport for this machine")),
  download: (machine, remote, local) => machine.provider === "namespace"
    ? checked(executable, ["download", machine.name, remote, local, "--mkdir"], { timeoutMs: 10 * 60_000 }).pipe(Effect.asVoid) : Effect.fail(failed("Wrong transport for this machine")),
  } satisfies WorkerTransport
}))
