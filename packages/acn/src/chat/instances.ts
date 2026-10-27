import { Deferred, type Duration, Effect, Either, ExecutionStrategy, Exit, Fiber, HashMap, Option, Scope, SynchronizedRef } from "effect"

// One open instance per key, opened on demand, shared by its holders, and closed once it is idle and
// unheld for its time to live; it stays the key's until it has closed, so a use meanwhile waits for
// the close and then opens afresh. An instance that fails, fails to open, or is removed is
// forgotten, so the key's next use opens afresh; a forgotten instance closes once no one holds it.
// Whoever forgets one holds it until it has done what follows, so it closes only after that.

/** What instances needs of an open instance: when it is idle, how to close it, and why it closed. */
export interface Instance {
  readonly awaitIdle: Effect.Effect<void>
  readonly close: (reason: "closed" | "removed") => Effect.Effect<void>
  /** Completes when the instance closes; `failed` when it closed because it failed. */
  readonly failed: Effect.Effect<boolean>
}

export interface Instances<Key, Runtime, E> {
  /** The key's instance, opened if needed, held until the scope closes. */
  readonly get: (key: Key) => Effect.Effect<Runtime, E, Scope.Scope>
  /**
   * Closes the key's instance as removed if it is open, then deletes what the key holds, as one
   * step that interrupting the caller does not abandon. Fails as the delete did.
   */
  readonly remove: (key: Key) => Effect.Effect<void, E>
  /** The key's instance if it is open, or once it opens if it is opening, without opening or holding it. */
  readonly current: (key: Key) => Effect.Effect<Option.Option<Runtime>>
}

/** One open of a key. `holders` and `idle` change only under the pool's lock. */
interface Entry<Key, Runtime, E> {
  readonly _tag: "Entry"
  readonly key: Key
  readonly opened: Deferred.Deferred<Runtime, E>
  /** The outcome of the delete of the removal that closed the instance. */
  readonly removed: Deferred.Deferred<void, E>
  readonly scope: Scope.CloseableScope
  holders: number
  /** The pending close of an unheld instance in the pool. */
  idle: Option.Option<Fiber.RuntimeFiber<void, E>>
}

interface Removal<E> {
  readonly _tag: "Removal"
  readonly done: Deferred.Deferred<void, E>
}

/** The close of an idle instance, under way. */
interface Closing {
  readonly _tag: "Closing"
  readonly closed: Deferred.Deferred<void>
}

export const makeInstances = <Key, Runtime extends Instance, E, R>(config: {
  /** Opens the key's instance; `removed` completes as the delete of a removal that closed it did. */
  readonly open: (key: Key, removed: Effect.Effect<void, E>) => Effect.Effect<Runtime, E, Scope.Scope | R>
  readonly remove: (key: Key) => Effect.Effect<void, E, R>
  /** Runs once an instance that failed is forgotten, so the next use opens the key afresh. */
  readonly failed: (key: Key) => Effect.Effect<void, never, R>
  readonly idleTimeToLive: Duration.DurationInput
}): Effect.Effect<Instances<Key, Runtime, E>, never, Scope.Scope | R> => Effect.gen(function* () {
  type Pool = HashMap.HashMap<Key, Entry<Key, Runtime, E> | Removal<E> | Closing>
  /** An entry, or what to wait for instead. */
  type Taken<A> = Either.Either<A, Effect.Effect<void, E>>
  const context = yield* Effect.context<R>()
  const scope = yield* Effect.scope
  const pool = yield* SynchronizedRef.make<Pool>(HashMap.empty())

  /** Runs `step` under the lock, then what it returns to run after. */
  const transition = (step: (pool: Pool) => Effect.Effect<readonly [Effect.Effect<void>, Pool]>) =>
    Effect.flatten(SynchronizedRef.modifyEffect(pool, step))

  const inPool = (current: Pool, entry: Entry<Key, Runtime, E>) =>
    Option.exists(HashMap.get(current, entry.key), (slot) => slot === entry)

  /** One more holder, which cancels a pending idle close. Under the lock. */
  const hold = (entry: Entry<Key, Runtime, E>) => Effect.suspend(() => {
    const idle = entry.idle
    entry.holders++
    entry.idle = Option.none()
    return Option.match(idle, { onNone: () => Effect.void, onSome: (fiber) => Effect.asVoid(Fiber.interrupt(fiber)) })
  })

  /** Takes the entry out of the pool if it is still there, held until its forgetter releases it. */
  const forget = (entry: Entry<Key, Runtime, E>) => transition((current) =>
    Effect.as(hold(entry), [Effect.void, inPool(current, entry) ? HashMap.remove(current, entry.key) : current] as const))

  // Once idle and then unheld for its time to live, the entry is replaced by its close, which leaves
  // the pool once the instance has closed. Every hold interrupts this under the lock, so it closes
  // only if none came first; once the close has replaced the entry, no hold reaches it, and it runs
  // to the end.
  const idle = (entry: Entry<Key, Runtime, E>) => Effect.uninterruptible(Effect.interruptible(
    Deferred.await(entry.opened).pipe(Effect.flatMap((runtime) => runtime.awaitIdle), Effect.zipRight(Effect.sleep(config.idleTimeToLive)))
  ).pipe(
    Effect.zipRight(Effect.interruptible(SynchronizedRef.modifyEffect(pool, (current) => Effect.map(Deferred.make<void>(), (closed) => {
      entry.idle = Option.none()
      return [closed, HashMap.set(current, entry.key, { _tag: "Closing", closed } satisfies Closing)] as const
    })))),
    Effect.flatMap((closed) => Scope.close(entry.scope, Exit.void).pipe(
      Effect.zipRight(SynchronizedRef.update(pool, HashMap.remove(entry.key))),
      Effect.zipRight(Deferred.succeed(closed, undefined)),
      Effect.asVoid
    ))
  ))

  /** One holder fewer: the last one closes a forgotten entry, or starts an entry's idle close. */
  const release = (entry: Entry<Key, Runtime, E>) => transition((current) => Effect.suspend(() => {
    entry.holders--
    if (entry.holders > 0) return Effect.succeed([Effect.void, current] as const)
    if (!inPool(current, entry)) return Effect.succeed([Scope.close(entry.scope, Exit.void), current] as const)
    return Effect.map(Effect.forkIn(idle(entry), scope), (fiber) => {
      entry.idle = Option.some(fiber)
      return [Effect.void, current] as const
    })
  }))

  // Runs apart from any holder, so a holder leaving never abandons an open others share. An open
  // that failed is forgotten, and closed if unheld, before anyone learns of it.
  const open = (entry: Entry<Key, Runtime, E>) => Effect.gen(function* () {
    const opened = yield* Effect.exit(Scope.extend(config.open(entry.key, Deferred.await(entry.removed)), entry.scope).pipe(
      Effect.provide(context)
    ))
    if (Exit.isFailure(opened)) {
      yield* forget(entry)
      yield* release(entry)
      yield* Deferred.done(entry.opened, opened)
      return
    }
    yield* Deferred.succeed(entry.opened, opened.value)
    if (!(yield* opened.value.failed)) return
    yield* forget(entry)
    yield* Effect.provide(config.failed(entry.key), context)
    yield* release(entry)
  })

  /** Holds the key's entry, made and opening if it has none, or gives the removal or close to wait for first. */
  const admit = (key: Key) => SynchronizedRef.modifyEffect(pool, (current): Effect.Effect<readonly [Taken<Entry<Key, Runtime, E>>, Pool]> => Effect.gen(function* () {
    const slot = HashMap.get(current, key)
    if (Option.isSome(slot)) {
      if (slot.value._tag === "Removal") return [Either.left(Deferred.await(slot.value.done)), current] as const
      if (slot.value._tag === "Closing") return [Either.left(Deferred.await(slot.value.closed)), current] as const
      yield* hold(slot.value)
      return [Either.right(slot.value), current] as const
    }
    const entry: Entry<Key, Runtime, E> = {
      _tag: "Entry",
      key,
      opened: yield* Deferred.make<Runtime, E>(),
      removed: yield* Deferred.make<void, E>(),
      scope: yield* Scope.fork(scope, ExecutionStrategy.sequential),
      holders: 1,
      idle: Option.none(),
    }
    yield* Effect.forkIn(Effect.interruptible(open(entry)), scope)
    return [Either.right(entry), HashMap.set(current, key, entry)] as const
  }))

  // A use during the key's removal or close waits for it to end, whatever its outcome, then opens afresh.
  const get = (key: Key): Effect.Effect<Runtime, E, Scope.Scope> => Effect.uninterruptibleMask((restore) =>
    Effect.flatMap(admit(key), Either.match({
      onLeft: (ended) => restore(Effect.zipRight(Effect.exit(ended), get(key))),
      onRight: (entry) => Effect.zipRight(Effect.addFinalizer(() => release(entry)), restore(Deferred.await(entry.opened))),
    })))

  const remove = (key: Key): Effect.Effect<void, E> => Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
    const removal: Removal<E> = { _tag: "Removal", done: yield* Deferred.make<void, E>() }
    // The key's entry, held and replaced by this removal; a removal already under way, whose outcome
    // is this one's; or a close under way, after which this removal starts again, as part of its step.
    const taken = yield* SynchronizedRef.modifyEffect(pool, (current): Effect.Effect<readonly [Taken<Option.Option<Entry<Key, Runtime, E>>>, Pool]> => Option.match(HashMap.get(current, key), {
      onNone: () => Effect.succeed([Either.right(Option.none()), HashMap.set(current, key, removal)] as const),
      onSome: (slot) => {
        switch (slot._tag) {
          case "Removal":
            return Effect.succeed([Either.left(restore(Deferred.await(slot.done))), current] as const)
          case "Closing":
            return Effect.succeed([Either.left(Effect.zipRight(Deferred.await(slot.closed), remove(key))), current] as const)
          case "Entry":
            return Effect.as(hold(slot), [Either.right(Option.some(slot)), HashMap.set(current, key, removal)] as const)
        }
      },
    }))
    if (Either.isLeft(taken)) return yield* taken.left
    const entry = taken.right
    if (Option.isSome(entry)) {
      const opened = yield* Effect.exit(Deferred.await(entry.value.opened))
      if (Exit.isSuccess(opened)) yield* opened.value.close("removed")
    }
    const deleted = yield* Effect.exit(Effect.provide(config.remove(key), context))
    if (Option.isSome(entry)) {
      yield* Deferred.done(entry.value.removed, deleted)
      yield* release(entry.value)
    }
    yield* SynchronizedRef.update(pool, HashMap.remove(key))
    yield* Deferred.done(removal.done, deleted)
    return yield* deleted
  }))

  return {
    get,
    current: (key) => Effect.flatMap(SynchronizedRef.get(pool), (current) => Option.match(HashMap.get(current, key), {
      onNone: () => Effect.succeedNone,
      // One that fails to open never is current.
      onSome: (slot) => slot._tag === "Entry" ? Effect.option(Deferred.await(slot.opened)) : Effect.succeedNone,
    })),
    remove,
  }
})
