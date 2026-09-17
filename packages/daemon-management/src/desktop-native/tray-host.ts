import { Context, Deferred, Effect, Layer, Option, Queue, Runtime, Schema, Stream, SubscriptionRef } from "effect"
import { DBusError, sessionBus, type Message } from "dbus-native"

export const TrayHostOwner = Schema.String.pipe(Schema.pattern(/^:[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+$/), Schema.maxLength(255), Schema.brand("TrayHostOwner"))
export const TrayHostState = Schema.Union(
  Schema.TaggedStruct("Available", { owner: TrayHostOwner }),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
)
export type TrayHostState = typeof TrayHostState.Type
class TrayBusUnavailable extends Schema.TaggedError<TrayBusUnavailable>()("TrayBusUnavailable", { message: Schema.String }) {}
class WatcherAbsent extends Schema.TaggedError<WatcherAbsent>()("WatcherAbsent", {}) {}
const watcher = "org.kde.StatusNotifierWatcher"
const watcherPath = "/StatusNotifierWatcher"
const busName = "org.freedesktop.DBus"
const busPath = "/org/freedesktop/DBus"
const unavailable = (message: string): TrayHostState => ({ _tag: "Unavailable", message })

/** One session connection. Signals invalidate an authoritative owner/property snapshot. */
export const observeLinuxTrayHost = (address: string) => Stream.unwrapScoped(Effect.gen(function* () {
  const invalidations = yield* Queue.sliding<void>(1)
  const disconnected = yield* Deferred.make<never, TrayBusUnavailable>()
  const runtime = yield* Effect.runtime<never>()
  const dispatch = Runtime.runFork(runtime)
  const bus = yield* Effect.acquireRelease(Effect.try({
    try: () => sessionBus({ busAddress: address, authMethods: ["EXTERNAL"], timeout: 2000, maxMessageSize: 1024 * 1024 }),
    catch: () => new TrayBusUnavailable({ message: "Could not connect to the desktop session bus." }),
  }), bus => Effect.promise(() => bus.close()).pipe(Effect.timeout("2 seconds"),
    Effect.catchAll(() => Effect.sync(() => { bus.connection.stream.destroy() }))))
  const onDisconnected = () => { dispatch(Deferred.fail(disconnected, new TrayBusUnavailable({ message: "The desktop session bus disconnected." }))) }
  // Keep an error listener until close: late socket errors must never escape into Electron main.
  bus.on("error", onDisconnected)
  bus.connection.on("error", onDisconnected)
  bus.connection.on("close", onDisconnected)
  const onMessage = (message: Message) => {
    if (message.type !== 4) return
    if ((message.sender === busName && message.path === busPath && message.interface === busName && message.member === "NameOwnerChanged" && message.body?.[0] === watcher) ||
        (message.path === watcherPath && message.interface === watcher && ["StatusNotifierHostRegistered", "StatusNotifierHostUnregistered"].includes(message.member ?? "")) ||
        (message.path === watcherPath && message.interface === "org.freedesktop.DBus.Properties" && message.member === "PropertiesChanged" && message.body?.[0] === watcher)) {
      dispatch(Queue.offer(invalidations, undefined))
    }
  }
  bus.connection.on("message", onMessage)
  yield* Effect.addFinalizer(() => Effect.sync(() => bus.connection.removeListener("message", onMessage)))
  // Match installation is acknowledged before the initial read, including an initially absent watcher.
  for (const rule of [
    `type='signal',sender='${busName}',interface='${busName}',path='${busPath}',member='NameOwnerChanged',arg0='${watcher}'`,
    `type='signal',sender='${watcher}',interface='${watcher}',path='${watcherPath}'`,
    `type='signal',sender='${watcher}',interface='org.freedesktop.DBus.Properties',path='${watcherPath}',member='PropertiesChanged',arg0='${watcher}'`,
  ]) yield* Effect.tryPromise({ try: () => bus.watch(rule), catch: () => new TrayBusUnavailable({ message: "Could not observe the desktop tray host." }) })
  const invoke = (message: Message) => Effect.tryPromise({
    try: signal => Promise.resolve(bus.invoke<unknown>(message, { signal, timeout: 2000 })),
    catch: error => error instanceof DBusError && ["org.freedesktop.DBus.Error.NameHasNoOwner", "org.freedesktop.DBus.Error.ServiceUnknown"].includes(error.dbusName ?? error.name)
      ? new WatcherAbsent() : new TrayBusUnavailable({ message: "Could not read the desktop tray host." }),
  })
  const owner = invoke({ destination: busName, path: busPath, interface: busName, member: "GetNameOwner", signature: "s", body: [watcher] }).pipe(
    Effect.flatMap(Schema.decodeUnknown(TrayHostOwner)), Effect.map(Option.some),
    Effect.catchTag("WatcherAbsent", () => Effect.succeed(Option.none())),
    Effect.catchTag("ParseError", () => Effect.fail(new TrayBusUnavailable({ message: "The desktop tray host returned an invalid identity." }))),
  )
  const snapshot = Effect.gen(function* () {
    const before = yield* owner
    if (Option.isNone(before)) return Option.some(unavailable("Your desktop is not providing a supported tray host."))
    const registered = yield* invoke({ destination: before.value, path: watcherPath,
      interface: "org.freedesktop.DBus.Properties", member: "Get", signature: "ss", body: [watcher, "IsStatusNotifierHostRegistered"],
    }).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.Boolean)),
      Effect.catchTag("ParseError", () => Effect.fail(new TrayBusUnavailable({ message: "The desktop tray host returned an invalid status." }))),
      Effect.catchTag("WatcherAbsent", () => Effect.succeed(false)))
    const after = yield* owner
    // A queued owner-change signal triggers the next read; never publish a retired owner's reply.
    if (!Option.contains(after, before.value)) return Option.none<TrayHostState>()
    return Option.some<TrayHostState>(registered ? { _tag: "Available", owner: before.value }
      : unavailable("Your desktop's tray host is not ready."))
  })
  yield* Queue.offer(invalidations, undefined)
  const observations = Stream.fromQueue(invalidations).pipe(
    Stream.flatMap(() => Stream.fromEffect(snapshot), { switch: true }), Stream.filterMap(value => value),
    Stream.changesWith(Schema.equivalence(TrayHostState)),
  )
  return Stream.merge(observations, Stream.fromEffect(Deferred.await(disconnected)), { haltStrategy: "either" })
}))

export interface LinuxTrayHost {
  readonly state: Effect.Effect<TrayHostState>
  readonly changes: Stream.Stream<TrayHostState>
}
export const LinuxTrayHost = Context.GenericTag<LinuxTrayHost>("@magnitudedev/daemon-management/LinuxTrayHost")

/** Reconnect the bus on failure; never periodically recreate the tray or probe an unchanged host. */
export const linuxTrayHostLayer = (address: string | undefined = process.env.DBUS_SESSION_BUS_ADDRESS) => Layer.scoped(LinuxTrayHost, Effect.gen(function* () {
  const state = yield* SubscriptionRef.make<TrayHostState>(unavailable("Checking desktop tray availability…"))
  const observe = address ? observeLinuxTrayHost(address) : Stream.succeed(unavailable("No desktop session bus is available."))
  const worker = observe.pipe(Stream.runForEach(value => SubscriptionRef.set(state, value)),
    Effect.catchAll(error => SubscriptionRef.set(state, unavailable(error.message)).pipe(Effect.zipRight(Effect.sleep("5 seconds")))))
  // Successful observation is resident; only a disconnected/failed observation reaches the next run.
  yield* (address ? Effect.forever(worker) : worker).pipe(Effect.forkScoped)
  return LinuxTrayHost.of({ state: SubscriptionRef.get(state), changes: state.changes })
}))
