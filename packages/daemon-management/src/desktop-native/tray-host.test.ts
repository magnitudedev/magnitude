import { createBroker, createConnection, sessionBus, Variant, type Message, type MessageBus } from "dbus-native"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { spawn, type ChildProcess } from "node:child_process"
import { join } from "node:path"
import { Effect, Fiber, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { LinuxTrayHost, linuxTrayHostLayer, observeLinuxTrayHost, type TrayHostState } from "./tray-host"

const watcher = "org.kde.StatusNotifierWatcher"
const path = "/StatusNotifierWatcher"
const eventually = async (predicate: () => boolean, timeout = 4000) => {
  const deadline = Date.now() + timeout
  while (!predicate() && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10))
  expect(predicate()).toBe(true)
}
const startWatcher = async (address: string, registered: boolean) => {
  const bus = sessionBus({ busAddress: address, authMethods: ["EXTERNAL"], timeout: 1000 })
  bus.on("error", () => {})
  bus.connection.on("error", () => {})
  const value = { IsStatusNotifierHostRegistered: registered }
  bus.exportInterface(value, path, { name: watcher, properties: { IsStatusNotifierHostRegistered: "b" } })
  expect((await bus.ownName(watcher)).isPrimaryOwner).toBe(true)
  return { bus, set: (registered: boolean) => {
    value.IsStatusNotifierHostRegistered = registered
    bus.sendSignal(path, watcher, registered ? "StatusNotifierHostRegistered" : "StatusNotifierHostUnregistered")
  } }
}
// A wire-level watcher can delay Get while relinquishing its well-known name. Keeping the old
// connection alive makes its late reply a real stale response, not simply a closed socket.
const delayedWatcher = async (address: string) => {
  const connection = createConnection({ busAddress: address, authMethods: ["EXTERNAL"] })
  const pending = new Map<number, (message: Message) => void>()
  const reads: Message[] = []
  let serial = 1
  connection.on("error", () => {})
  connection.on("message", message => {
    if (message.replySerial !== undefined) pending.get(message.replySerial)?.(message)
    else if (message.type === 1 && message.member === "Get") reads.push(message)
  })
  const request = (member: string, signature = "", body: unknown[] = []) => new Promise<Message>((resolve, reject) => {
    const current = serial++
    const timeout = setTimeout(() => { pending.delete(current); reject(new Error(`No ${member} reply`)) }, 2000)
    pending.set(current, message => { clearTimeout(timeout); pending.delete(current); resolve(message) })
    connection.message({ type: 1, serial: current, destination: "org.freedesktop.DBus", path: "/org/freedesktop/DBus", interface: "org.freedesktop.DBus", member, signature, body })
  })
  await new Promise<void>((resolve, reject) => { connection.once("connect", resolve); connection.once("error", reject) })
  await request("Hello")
  await request("RequestName", "su", [watcher, 0])
  return { reads, release: () => request("ReleaseName", "s", [watcher]), close: () => connection[Symbol.asyncDispose](),
    reply: () => { for (const message of reads) connection.message({ type: 2, serial: serial++, destination: message.sender,
      replySerial: message.serial, signature: "v", body: [new Variant("b", false)],
    }) },
  }
}
const withBus = async (native: boolean, use: (address: string) => Promise<void>) => {
  const directory = await mkdtemp("/tmp/magnitude-tray-")
  const broker = native ? undefined : createBroker()
  let daemon: ChildProcess | undefined
  try {
    const address = native ? await (async () => {
      const config = join(directory, "bus.conf")
      await writeFile(config, `<busconfig><type>session</type><listen>unix:tmpdir=${directory}</listen><auth>EXTERNAL</auth><policy context="default"><allow user="*"/><allow own="*"/><allow send_destination="*"/><allow receive_sender="*"/></policy></busconfig>`)
      daemon = spawn(process.env.MAGNITUDE_TEST_DBUS_DAEMON!, ["--nofork", `--config-file=${config}`, "--print-address=1"], { stdio: ["ignore", "pipe", "inherit"] })
      return await new Promise<string>((resolve, reject) => {
        let output = ""
        const timeout = setTimeout(() => reject(new Error("Private D-Bus did not start")), 5000)
        daemon!.once("error", error => { clearTimeout(timeout); reject(error) })
        daemon!.stdout!.on("data", chunk => { output += chunk; if (output.includes("\n")) { clearTimeout(timeout); resolve(output.trim()) } })
      })
    })() : await new Promise<string>((resolve, reject) => broker!.listen({ socket: join(directory, "bus") }, (error, address) => error ? reject(error) : resolve(address)))
    await use(address)
  } finally {
    if (broker) await new Promise<void>(resolve => broker.close(resolve))
    if (daemon && daemon.exitCode === null && daemon.signalCode === null) {
      const exited = new Promise<void>(resolve => daemon!.once("exit", () => resolve()))
      daemon.kill("SIGTERM")
      await exited
    }
    await rm(directory, { recursive: true, force: true })
  }
}

describe.skipIf(process.platform === "win32")("Linux tray host observation", () => {
  it.each([false, ...(process.env.MAGNITUDE_TEST_DBUS_DAEMON ? [true] : [])])("recovers late host and watcher replacement (native daemon=%s)", async native => withBus(native, async address => {
    const values: TrayHostState[] = []
    const buses: MessageBus[] = []
    const observer = Effect.runFork(observeLinuxTrayHost(address).pipe(Stream.runForEach(value => Effect.sync(() => { values.push(value) }))))
    try {
      await eventually(() => values.at(-1)?._tag === "Unavailable")
      const first = await startWatcher(address, false)
      buses.push(first.bus)
      await eventually(() => values.at(-1)?._tag === "Unavailable" && values.length >= 2)
      first.set(true)
      await eventually(() => values.at(-1)?._tag === "Available")
      const available = values.at(-1)!
      expect(available).toMatchObject({ _tag: "Available", owner: first.bus.name })
      const count = values.length
      for (let index = 0; index < 20; index++) first.set(true)
      await new Promise(resolve => setTimeout(resolve, 60))
      expect(values.length).toBe(count)
      first.set(false)
      await eventually(() => values.at(-1)?._tag === "Unavailable")
      first.set(true)
      await eventually(() => values.at(-1)?._tag === "Available")
      await first.bus.close()
      await eventually(() => values.at(-1)?._tag === "Unavailable")
      const second = await startWatcher(address, true)
      buses.push(second.bus)
      await eventually(() => values.at(-1)?._tag === "Available")
      expect(values.at(-1)).toMatchObject({ _tag: "Available", owner: second.bus.name })
      expect(second.bus.name).not.toBe(first.bus.name)
      await Effect.runPromise(Fiber.interrupt(observer))
      const stoppedCount = values.length
      second.set(false)
      await new Promise(resolve => setTimeout(resolve, 30))
      expect(values.length).toBe(stoppedCount)
    } finally {
      await Effect.runPromise(Fiber.interrupt(observer))
      await Promise.all(buses.map(bus => bus.close()))
    }
  }), 15000)
  it("reports a missing session bus without inventing a tray host", async () => {
    const value = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const host = yield* LinuxTrayHost
      return yield* host.changes.pipe(Stream.filter(state => state._tag === "Unavailable" && state.message.includes("No desktop")), Stream.runHead)
    }).pipe(Effect.provide(linuxTrayHostLayer("")))))
    expect(value).toMatchObject({ _tag: "Some", value: { _tag: "Unavailable" } })
  })
  it("discards an old watcher's delayed property reply after ownership changes", async () => withBus(false, async address => {
    const old = await delayedWatcher(address)
    let replacement: Awaited<ReturnType<typeof startWatcher>> | undefined
    const values: TrayHostState[] = []
    const observer = Effect.runFork(observeLinuxTrayHost(address).pipe(Stream.runForEach(value => Effect.sync(() => { values.push(value) }))))
    try {
      await eventually(() => old.reads.length > 0)
      await old.release()
      replacement = await startWatcher(address, true)
      await eventually(() => values.at(-1)?._tag === "Available")
      const count = values.length
      old.reply()
      await new Promise(resolve => setTimeout(resolve, 60))
      expect(values).toHaveLength(count)
      expect(values.at(-1)).toMatchObject({ _tag: "Available", owner: replacement.bus.name })
    } finally {
      await Effect.runPromise(Fiber.interrupt(observer))
      await old.close()
      await replacement?.bus.close()
    }
  }))
  it("recovers a lost session connection and stops reconnecting when its scope closes", async () => {
    const directory = await mkdtemp("/tmp/magnitude-tray-reconnect-")
    let broker = createBroker()
    const listen = () => new Promise<string>((resolve, reject) => broker.listen({ socket: join(directory, "bus") }, (error, address) => error ? reject(error) : resolve(address)))
    const address = (await listen()).split(",guid=")[0]!
    const buses: MessageBus[] = []
    const values: TrayHostState[] = []
    const observer = Effect.runFork(Effect.scoped(Effect.gen(function* () {
      const host = yield* LinuxTrayHost
      yield* host.changes.pipe(Stream.runForEach(value => Effect.sync(() => { values.push(value) })))
    }).pipe(Effect.provide(linuxTrayHostLayer(address)))))
    try {
      const first = await startWatcher(address, true)
      buses.push(first.bus)
      await eventually(() => values.at(-1)?._tag === "Available")
      await first.bus.close()
      await new Promise<void>(resolve => broker.close(resolve))
      await eventually(() => values.at(-1)?._tag === "Unavailable")
      broker = createBroker()
      await listen()
      const second = await startWatcher(address, true)
      buses.push(second.bus)
      await eventually(() => values.at(-1)?._tag === "Available", 8000)
      await Effect.runPromise(Fiber.interrupt(observer))
      // Only the test watcher remains connected; the observer's name and match rules are gone.
      await eventually(() => broker.names().filter(name => name.startsWith(":")).length === 1)
      expect(broker.names().filter(name => name.startsWith(":"))).toEqual([second.bus.name])
    } finally {
      await Effect.runPromise(Fiber.interrupt(observer))
      await Promise.all(buses.map(bus => bus.close()))
      await new Promise<void>(resolve => broker.close(resolve))
      await rm(directory, { recursive: true, force: true })
    }
  }, 15000)
})
