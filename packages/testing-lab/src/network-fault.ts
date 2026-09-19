import { Context, Effect, Layer, Option, Schema, Scope } from "effect"
import { randomUUID } from "node:crypto"
import { lookup } from "node:dns/promises"
import { createConnection } from "node:net"
import { userInfo } from "node:os"
import { DisposableDesktopUser } from "./desktop-environment"
import { InfrastructureFailure } from "./domain"
import { command, ProcessExecutor } from "./process"

export const NetworkRuleId = Schema.String.pipe(Schema.pattern(/^magnitude_lab_[a-f0-9]{32}$/), Schema.brand("NetworkRuleId"))
export const NetworkIsolation = Schema.Struct({ rule: NetworkRuleId, uid: Schema.Int.pipe(Schema.positive()) })
export const NetworkProbeAddress = Schema.Struct({ address: Schema.NonEmptyString, family: Schema.Literal(4, 6), port: Schema.Int.pipe(Schema.between(1, 65535)) })
export interface NetworkFault {
  readonly resolve: Effect.Effect<typeof NetworkProbeAddress.Type, InfrastructureFailure>
  readonly reachable: (address: typeof NetworkProbeAddress.Type) => Effect.Effect<boolean>
  readonly isolate: (onCleanupError: (message: string) => void) => Effect.Effect<typeof NetworkIsolation.Type, InfrastructureFailure, Scope.Scope>
}
export const NetworkFault = Context.GenericTag<NetworkFault>("@magnitudedev/testing-lab/NetworkFault")
const fail = (message: string) => new InfrastructureFailure({ operation: "network-fault", message })

/** The address is resolved before isolation and reused, so DNS failure cannot masquerade as an offline proof. */
export const resolveNetworkProbe = Effect.tryPromise({ try: () => lookup("huggingface.co", { family: 4 }),
  catch: () => fail("Cannot resolve the model host for the external network control") }).pipe(
  Effect.map(result => NetworkProbeAddress.make({ address: result.address, family: 4, port: 443 })),
  Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => fail("External network control resolution timed out") }))

/** One fresh direct TCP connection; no proxy, DNS, HTTP cache or reused connection. */
export const networkReachable = (address: typeof NetworkProbeAddress.Type) => Effect.async<boolean>(resume => {
  const socket = createConnection({ host: address.address, port: address.port, family: address.family })
  let finished = false
  const finish = (reachable: boolean) => { if (!finished) { finished = true; socket.destroy(); resume(Effect.succeed(reachable)) } }
  socket.once("connect", () => finish(true))
  socket.once("error", () => finish(false))
  socket.setTimeout(5000, () => finish(false))
  return Effect.sync(() => { finished = true; socket.destroy() })
})
// One inet table covers IPv4 and IPv6. It never changes another table or host policy.
export const isolationRules = (isolation: typeof NetworkIsolation.Type) => `table inet ${isolation.rule} {\n chain output {\n  type filter hook output priority -10; policy accept;\n  meta skuid ${isolation.uid} oifname != "lo" counter reject\n }\n}\n`
export const linuxNetworkFault = Layer.effect(NetworkFault, Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const user = yield* Effect.serviceOption(DisposableDesktopUser)
  const nft = (args: readonly string[], stdin = Option.none<string>()) => command("/usr/bin/sudo", ["-n", "/usr/sbin/nft", ...args], {
    stdin, inheritEnv: false, env: { PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }, timeoutMs: 15000, maxOutputBytes: 16384,
  }).pipe(Effect.provideService(ProcessExecutor, executor), Effect.flatMap(result => result.exitCode === 0 ? Effect.void : Effect.fail(fail(result.stderr.trim().slice(-2000)))))
  return { resolve: resolveNetworkProbe, reachable: networkReachable, isolate: onCleanupError => Effect.gen(function* () {
    if (process.platform !== "linux" || Option.isNone(user)) return yield* fail("Offline network isolation requires a qualified disposable Linux user")
    const identity = yield* Effect.try({ try: () => userInfo(), catch: () => fail("Cannot inspect the network fault user") })
    if (identity.uid <= 0 || identity.homedir !== user.value.home) return yield* fail("Network fault user does not match the disposable desktop account")
    const isolation = NetworkIsolation.make({ uid: identity.uid, rule: NetworkRuleId.make(`magnitude_lab_${randomUUID().replaceAll("-", "")}`) })
    // Validate support and permissions before registering a potentially destructive operation.
    yield* nft(["--check", "destroy", "table", "inet", isolation.rule])
    yield* nft(["--check", "--file", "-"], Option.some(isolationRules(isolation)))
    // Register cleanup before installation: cancellation or a lost subprocess response may leave a committed table.
    yield* Effect.addFinalizer(() => nft(["destroy", "table", "inet", isolation.rule]).pipe(Effect.catchAll(error => Effect.sync(() => { onCleanupError(error.message) }))))
    yield* nft(["--file", "-"], Option.some(isolationRules(isolation)))
    return isolation
  }) } satisfies NetworkFault
}))
