import { Effect, Layer } from "effect"
import { createServer, type AddressInfo } from "node:net"
import { expect, test } from "vitest"
import { linuxNetworkFault, NetworkFault, NetworkProbeAddress, networkReachable } from "../src/network-fault"
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
    Effect.provide(linuxNetworkFault.pipe(Layer.provide(Layer.succeed(ProcessExecutor, { run: () => {
      commands++
      return Effect.die("Unexpected host firewall command")
    } })))), Effect.either))
  expect(result._tag).toBe("Left")
  expect(commands).toBe(0)
})
