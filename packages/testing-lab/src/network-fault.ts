import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option, Schema, Scope } from "effect"
import { randomUUID } from "node:crypto"
import { lookup } from "node:dns/promises"
import { createConnection, isIP } from "node:net"
import { userInfo } from "node:os"
import { win32 } from "node:path"
import { DisposableDesktopUser } from "./desktop-environment"
import { InfrastructureFailure } from "./domain"
import { command, ProcessExecutor } from "./process"

export const NetworkRuleId = Schema.String.pipe(Schema.pattern(/^magnitude_lab_[a-f0-9]{32}$/), Schema.brand("NetworkRuleId"))
const IpAddress = Schema.String.pipe(Schema.filter(value => isIP(value) !== 0))
export const NetworkProbeAddress = Schema.Struct({ address: IpAddress, family: Schema.Literal(4, 6), port: Schema.Int.pipe(Schema.between(1, 65535)) })
const LinuxNetworkIsolation = Schema.TaggedStruct("linux", { rule: NetworkRuleId, uid: Schema.Int.pipe(Schema.positive()),
  controlPlane: Schema.Array(NetworkProbeAddress), resolvers: Schema.Array(IpAddress) })
const WindowsAddressRange = Schema.String.pipe(Schema.pattern(/^\d{1,3}(?:\.\d{1,3}){3}-\d{1,3}(?:\.\d{1,3}){3}$/))
const WindowsNetworkIsolation = Schema.TaggedStruct("windows", { rule: NetworkRuleId,
  controlPlane: Schema.Array(NetworkProbeAddress), resolvers: Schema.Array(IpAddress), ranges: Schema.Array(WindowsAddressRange),
  externalIpv6Reachable: Schema.Literal(false) })
export const NetworkIsolation = Schema.Union(LinuxNetworkIsolation, WindowsNetworkIsolation)
/** Trusted worker bootstrap authority, never supplied by the candidate or a test request. */
export interface NetworkControlPlane { readonly origin: string }
export const NetworkControlPlane = Context.GenericTag<NetworkControlPlane>("@magnitudedev/testing-lab/NetworkControlPlane")
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
export const isolationRules = (isolation: typeof LinuxNetworkIsolation.Type) => {
  const user = `meta skuid ${isolation.uid} oifname != "lo"`
  const allowed = [
    ...isolation.controlPlane.map(endpoint => `  ${user} ${endpoint.family === 6 ? "ip6" : "ip"} daddr ${endpoint.address} tcp dport ${endpoint.port} accept`),
    ...isolation.resolvers.flatMap(address => ["tcp", "udp"].map(protocol => `  ${user} ${isIP(address) === 6 ? "ip6" : "ip"} daddr ${address} ${protocol} dport 53 accept`)),
  ]
  return [`table inet ${isolation.rule} {`, " chain output {", "  type filter hook output priority -10; policy accept;", ...allowed,
    `  ${user} meta l4proto tcp counter reject with tcp reset`, `  ${user} counter reject`, " }", "}", ""].join("\n")
}

export const controlPlaneAddresses = (origin: string) => Effect.gen(function* () {
  const url = yield* Effect.try({ try: () => new URL(origin), catch: () => fail("Invalid lab control origin") })
  if (url.username || url.password || url.pathname !== "/" || url.search || url.hash ||
    (url.protocol !== "https:" && !(url.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)))) return yield* fail("Invalid lab control origin")
  const addresses = yield* Effect.tryPromise({ try: () => lookup(url.hostname.replace(/^\[|\]$/g, ""), { all: true }),
    catch: () => fail("Cannot resolve lab control addresses before isolation") }).pipe(
      Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => fail("Lab control resolution timed out") }))
  if (!addresses.length || addresses.length > 16) return yield* fail("Unexpected number of lab control addresses")
  const controlPlane = yield* Schema.decodeUnknown(Schema.Array(NetworkProbeAddress))(addresses.map(address => ({ ...address,
    port: Number(url.port || (url.protocol === "https:" ? 443 : 80)) }))).pipe(Effect.mapError(() => fail("Invalid resolved lab control address")))
  return controlPlane
})
export const controlPlaneExceptions = (origin: string) => Effect.gen(function* () {
  const controlPlane = yield* controlPlaneAddresses(origin)
  const text = yield* (yield* FileSystem.FileSystem).readFileString("/etc/resolv.conf").pipe(Effect.mapError(() => fail("Cannot read worker DNS configuration")))
  const resolvers = [...new Set(text.split("\n").flatMap(line => /^\s*nameserver\s+([^\s#;]+)/.exec(line)?.[1] ?? []))]
  if (resolvers.length > 8 || resolvers.some(address => !isIP(address))) return yield* fail("Invalid worker DNS configuration")
  return { controlPlane, resolvers }
})
export const linuxNetworkFault = Layer.effect(NetworkFault, Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const fs = yield* FileSystem.FileSystem
  const control = yield* Effect.serviceOption(NetworkControlPlane)
  const user = yield* Effect.serviceOption(DisposableDesktopUser)
  const nft = (args: readonly string[], stdin = Option.none<string>()) => command("/usr/bin/sudo", ["-n", "/usr/sbin/nft", ...args], {
    stdin, inheritEnv: false, env: { PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }, timeoutMs: 15000, maxOutputBytes: 16384,
  }).pipe(Effect.provideService(ProcessExecutor, executor), Effect.flatMap(result => result.exitCode === 0 ? Effect.void : Effect.fail(fail(result.stderr.trim().slice(-2000)))))
  return { resolve: resolveNetworkProbe, reachable: networkReachable, isolate: onCleanupError => Effect.gen(function* () {
    if (process.platform !== "linux" || Option.isNone(user)) return yield* fail("Offline network isolation requires a qualified disposable Linux user")
    const identity = yield* Effect.try({ try: () => userInfo(), catch: () => fail("Cannot inspect the network fault user") })
    if (identity.uid <= 0 || identity.homedir !== user.value.home) return yield* fail("Network fault user does not match the disposable desktop account")
    const exceptions = Option.isSome(control) ? yield* controlPlaneExceptions(control.value.origin).pipe(Effect.provideService(FileSystem.FileSystem, fs))
      : { controlPlane: [], resolvers: [] }
    const isolation = LinuxNetworkIsolation.make({ _tag: "linux", uid: identity.uid,
      rule: NetworkRuleId.make(`magnitude_lab_${randomUUID().replaceAll("-", "")}`), ...exceptions })
    // Validate support and permissions before registering a potentially destructive operation.
    yield* nft(["--check", "destroy", "table", "inet", isolation.rule])
    yield* nft(["--check", "--file", "-"], Option.some(isolationRules(isolation)))
    // Register cleanup before installation: cancellation or a lost subprocess response may leave a committed table.
    yield* Effect.addFinalizer(() => nft(["destroy", "table", "inet", isolation.rule]).pipe(Effect.catchAll(error => Effect.sync(() => { onCleanupError(error.message) }))))
    yield* nft(["--file", "-"], Option.some(isolationRules(isolation)))
    return isolation
  }) } satisfies NetworkFault
}))

const ipv4Number = (address: string) => address.split(".").reduce((value, part) => value * 256 + Number(part), 0)
const ipv4String = (value: number) => [24, 16, 8, 0].map(shift => (value >>> shift) & 255).join(".")
/** Cover every IPv4 destination except exact trusted endpoints and the loopback block. */
export const windowsIsolationRanges = (allowed: readonly string[]) => {
  const excluded = [...new Set([...allowed.filter(address => isIP(address) === 4), "168.63.129.16", "169.254.169.254"]
    .map(ipv4Number))].sort((a, b) => a - b)
  const ranges: string[] = []
  let start = 0
  for (const value of excluded) {
    if (start < value) ranges.push(`${ipv4String(start)}-${ipv4String(value - 1)}`)
    start = value + 1
  }
  if (start <= 0xffffffff) ranges.push(`${ipv4String(start)}-255.255.255.255`)
  return ranges
}

const windowsPowerShell = (script: string, environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const systemRoot = process.env.SystemRoot ?? process.env.WINDIR
  if (!systemRoot || !win32.isAbsolute(systemRoot)) return yield* fail("Windows network isolation requires the native system root")
  const result = yield* command(win32.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
    ["-NoProfile", "-NonInteractive", "-Command", script], { inheritEnv: false, env: {
      SystemRoot: systemRoot, WINDIR: systemRoot, PATH: `${systemRoot}\\System32;${systemRoot}`,
      ...environment,
    }, timeoutMs: 30_000, maxOutputBytes: 64 * 1024 }).pipe(Effect.provideService(ProcessExecutor, executor))
  if (result.exitCode !== 0) return yield* fail((result.stderr || result.stdout).trim().slice(-2000) || "Windows firewall command failed")
  return result.stdout
})

const windowsResolvers = windowsPowerShell(String.raw`$ErrorActionPreference='Stop'
$addresses=@(Get-DnsClientServerAddress -AddressFamily IPv4 | ForEach-Object {$_.ServerAddresses} | Where-Object {$_} | Sort-Object -Unique)
ConvertTo-Json -InputObject @($addresses) -Compress`, {}).pipe(
  Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Array(IpAddress)))),
  Effect.mapError(() => fail("Cannot inspect Windows DNS resolver addresses")))

const externalIpv6Reachable = Effect.tryPromise({ try: () => lookup("huggingface.co", { family: 6 }), catch: () => fail("Cannot resolve external IPv6 control") }).pipe(
  Effect.catchAll(() => Effect.succeed(null)),
  Effect.flatMap(result => result === null ? Effect.succeed(false)
    : networkReachable(NetworkProbeAddress.make({ address: result.address, family: 6, port: 443 }))))

/** Windows workers are whole disposable VMs, so the temporary rule owns the guest's outbound IPv4 policy. */
export const windowsNetworkFault = Layer.effect(NetworkFault, Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const control = yield* Effect.serviceOption(NetworkControlPlane)
  const user = yield* Effect.serviceOption(DisposableDesktopUser)
  const powershell = (script: string, environment: Readonly<Record<string, string>>) => windowsPowerShell(script, environment).pipe(
    Effect.provideService(ProcessExecutor, executor))
  return { resolve: resolveNetworkProbe, reachable: networkReachable, isolate: onCleanupError => Effect.gen(function* () {
    if (process.platform !== "win32" || Option.isNone(user)) return yield* fail("Offline network isolation requires a qualified disposable Windows user")
    if (yield* externalIpv6Reachable) return yield* fail("Windows worker has external IPv6 connectivity that the qualified isolation profile does not cover")
    const controlPlane = Option.isSome(control) ? yield* controlPlaneAddresses(control.value.origin) : []
    const resolvers = yield* windowsResolvers.pipe(Effect.provideService(ProcessExecutor, executor))
    const ranges = windowsIsolationRanges([...controlPlane.map(endpoint => endpoint.address), ...resolvers, "127.0.0.1"])
    if (!ranges.length || ranges.length > 64) return yield* fail("Windows network isolation produced an invalid address range set")
    const isolation = WindowsNetworkIsolation.make({ _tag: "windows", rule: NetworkRuleId.make(`magnitude_lab_${randomUUID().replaceAll("-", "")}`),
      controlPlane, resolvers, ranges, externalIpv6Reachable: false })
    const common = { LAB_NETWORK_RULE: isolation.rule, LAB_NETWORK_RANGES: isolation.ranges.join(",") }
    yield* powershell(String.raw`$ErrorActionPreference='Stop'
if (Get-NetFirewallRule -Name $env:LAB_NETWORK_RULE -ErrorAction SilentlyContinue) {throw 'Network isolation rule already exists'}
New-NetFirewallRule -Name $env:LAB_NETWORK_RULE -DisplayName $env:LAB_NETWORK_RULE -Direction Outbound -Action Block -Enabled True -Profile Any -RemoteAddress ($env:LAB_NETWORK_RANGES -split ',') | Out-Null
if (-not (Get-NetFirewallRule -Name $env:LAB_NETWORK_RULE -ErrorAction Stop).Enabled) {throw 'Network isolation rule is not enabled'}`, common)
    yield* Effect.addFinalizer(() => powershell(String.raw`$ErrorActionPreference='Stop'
$rule=Get-NetFirewallRule -Name $env:LAB_NETWORK_RULE -ErrorAction SilentlyContinue
if ($rule) {$rule | Remove-NetFirewallRule -ErrorAction Stop}`, { LAB_NETWORK_RULE: isolation.rule }).pipe(
      Effect.catchAll(error => Effect.sync(() => { onCleanupError(error.message) }))))
    return isolation
  }) } satisfies NetworkFault
}))

export const nativeNetworkFault = process.platform === "win32" ? windowsNetworkFault : linuxNetworkFault
