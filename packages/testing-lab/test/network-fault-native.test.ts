import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { createServer, type AddressInfo } from "node:net"
import { userInfo } from "node:os"
import { expect, test } from "vitest"
import { DisposableDesktopUser } from "../src/desktop-environment"
import { AssertionFailure } from "../src/domain"
import { linuxNetworkFault, NetworkFault, NetworkProbeAddress } from "../src/network-fault"
import { ProcessExecutorLive } from "../src/process"

// Explicit opt-in is reserved for the disposable Azure guest, never a developer machine or Spark.
const enabled = process.platform === "linux" && process.env.LAB_NATIVE_NETWORK_FAULT === "1"
for (const injectFailure of [false, true]) test.skipIf(!enabled)(`native offline scope restores traffic after failure=${injectFailure}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const network = yield* NetworkFault
  const cleanup: string[] = []
  const address = yield* network.resolve
  expect(yield* network.reachable(address)).toBe(true)
  const server = yield* Effect.acquireRelease(Effect.promise(() => new Promise<ReturnType<typeof createServer>>(resolve => {
    const server = createServer(socket => socket.end())
    server.listen(0, "127.0.0.1", () => resolve(server))
  })), server => Effect.promise(() => new Promise<void>(resolve => server.close(() => resolve()))))
  const local = NetworkProbeAddress.make({ address: "127.0.0.1", family: 4, port: (server.address() as AddressInfo).port })
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const receipt = yield* network.isolate(message => { cleanup.push(message) })
    expect(receipt.uid).toBe(userInfo().uid)
    expect(yield* network.reachable(address)).toBe(false)
    expect(yield* network.reachable(local)).toBe(true)
    if (injectFailure) return yield* new AssertionFailure({ message: "Intentional offline scenario failure" })
  })).pipe(Effect.either)
  expect(result._tag).toBe(injectFailure ? "Left" : "Right")
  expect(cleanup).toEqual([])
  expect(yield* network.reachable(address)).toBe(true)
})).pipe(Effect.provide(linuxNetworkFault.pipe(Layer.provide([
  BunContext.layer, ProcessExecutorLive, Layer.succeed(DisposableDesktopUser, { home: userInfo().homedir }),
]))))), 30000)
