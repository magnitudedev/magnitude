import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"
import { createServer, type AddressInfo } from "node:net"
import { expect, test } from "vitest"
import { controlPlaneExceptions, isolationRules, linuxNetworkFault, NetworkFault, NetworkIsolation, NetworkProbeAddress, networkReachable } from "../src/network-fault"
import { ProcessExecutor } from "../src/process"

test("network probes establish a fresh TCP connection and reject a closed port", async () => {
  const server = createServer(socket => socket.end())
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve))
  const address = NetworkProbeAddress.make({ address: "127.0.0.1", family: 4, port: (server.address() as AddressInfo).port })
  try { expect(await Effect.runPromise(networkReachable(address))).toBe(true) }
  finally { await new Promise<void>(resolve => server.close(() => resolve())) }
  expect(await Effect.runPromise(networkReachable(address))).toBe(false)
})

test("network isolation is unavailable without the qualified disposable user capability", async () => {
  let commands = 0
  const result = await Effect.runPromise(Effect.scoped(Effect.flatMap(NetworkFault, network => network.isolate(() => {}))).pipe(
    Effect.provide(linuxNetworkFault.pipe(Layer.provide(Layer.merge(BunContext.layer, Layer.succeed(ProcessExecutor, { run: () => {
      commands++
      return Effect.die("Unexpected host firewall command")
    } }))))), Effect.either))
  expect(result._tag).toBe("Left")
  expect(commands).toBe(0)
})

test("offline evidence records bounded control and DNS exceptions without allowing model traffic", async () => {
  const exceptions = await Effect.runPromise(controlPlaneExceptions("https://127.0.0.1:8443").pipe(
    Effect.provide(FileSystem.layerNoop({ readFileString: () => Effect.succeed("nameserver 168.63.129.16 # Azure\nnameserver 168.63.129.16\nnameserver ::1\nsearch example.invalid\n") }))))
  expect(exceptions).toEqual({ controlPlane: [{ address: "127.0.0.1", family: 4, port: 8443 }], resolvers: ["168.63.129.16", "::1"] })
  const isolation = Schema.decodeUnknownSync(NetworkIsolation)({ uid: 1000, rule: `magnitude_lab_${"a".repeat(32)}`, ...exceptions })
  const rules = isolationRules(isolation)
  const allows = rules.split("\n").filter(line => line.endsWith(" accept"))
  expect(allows).toHaveLength(5)
  expect(allows.every(line => line.includes('meta skuid 1000 oifname != "lo"'))).toBe(true)
  expect(allows[0]).toContain("ip daddr 127.0.0.1 tcp dport 8443 accept")
  expect(allows.slice(1).every(line => line.endsWith("dport 53 accept"))).toBe(true)
  expect(rules.indexOf(allows[4]!)).toBeLessThan(rules.indexOf("counter reject"))
  expect(() => Schema.decodeUnknownSync(NetworkProbeAddress)({ address: "1.2.3.4; flush ruleset", family: 4, port: 443 })).toThrow()
})

test("control exceptions reject credential-bearing origins and malformed resolver addresses", async () => {
  for (const [origin, resolver] of [["https://user:secret@example.invalid", "1.2.3.4"], ["https://127.0.0.1", "not-an-address"]]) {
    const result = await Effect.runPromise(controlPlaneExceptions(origin!).pipe(
      Effect.provide(FileSystem.layerNoop({ readFileString: () => Effect.succeed(`nameserver ${resolver}\n`) })), Effect.either))
    expect(result._tag).toBe("Left")
  }
})
