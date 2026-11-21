import { Effect, Option, PubSub, Ref, Stream } from "effect"
import type { ApplicationAction, Page } from "./desktop-rpc"

type Action = typeof ApplicationAction.Type

/**
 * What the window's renderer is asked to do: actions published while it listens, and the latest page asked for (by
 * the tray or a menu), held until a renderer receives it so a cold start or reload never loses it, and given once.
 * With no page asked for, the renderer decides where it opens.
 */
export const makeHostActions = Effect.gen(function* () {
  const actions = yield* PubSub.unbounded<Action>()
  const pendingPage = yield* Ref.make(Option.none<Page>())
  const received = (action: Action) => action._tag === "Navigate" ? Ref.set(pendingPage, Option.none()) : Effect.void
  return {
    publish: (action: Action) => PubSub.publish(actions, action).pipe(Effect.asVoid),
    /** Asks for `page`: the renderer listening now is sent it, and one that subscribes before that gets it first. */
    navigate: (page: Page) => Ref.set(pendingPage, Option.some(page)).pipe(Effect.zipRight(PubSub.publish(actions, { _tag: "Navigate", page }))),
    /** Holds `page` for the next renderer without sending it, for a window that has none yet. */
    request: (page: Page) => Ref.set(pendingPage, Option.some(page)),
    subscribe: Stream.unwrapScoped(Effect.gen(function* () {
      const subscription = yield* PubSub.subscribe(actions)
      const page = yield* Ref.getAndSet(pendingPage, Option.none())
      const pending: ReadonlyArray<Action> = Option.toArray(Option.map(page, (value): Action => ({ _tag: "Navigate", page: value })))
      return Stream.concat(Stream.fromIterable(pending), Stream.fromQueue(subscription).pipe(Stream.tap(received)))
    })),
  }
})
