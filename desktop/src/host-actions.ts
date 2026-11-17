import { Effect, Option, PubSub, Ref, Stream } from "effect"
import type { ApplicationAction, Page } from "./desktop-rpc"

type Action = typeof ApplicationAction.Type
type OpenModel = Extract<Action, { readonly _tag: "OpenHuggingFaceModel" }>

/**
 * What the window's renderer is asked to do: actions published while it listens, and the latest page and model link
 * asked for (by the tray, a menu or a link), each held until a renderer receives it so a cold start or reload never
 * loses it, and given once. With no page asked for, the renderer decides where it opens.
 */
export const makeHostActions = Effect.gen(function* () {
  const actions = yield* PubSub.unbounded<Action>()
  const pendingPage = yield* Ref.make(Option.none<Page>())
  const pendingAddModel = yield* Ref.make(Option.none<OpenModel>())
  const received = (action: Action) => action._tag === "Navigate" ? Ref.set(pendingPage, Option.none())
    : action._tag === "OpenHuggingFaceModel" ? Ref.set(pendingAddModel, Option.none())
    : Effect.void
  return {
    publish: (action: Action) => PubSub.publish(actions, action).pipe(Effect.asVoid),
    /** Asks for `page`: the renderer listening now is sent it, and one that subscribes before that gets it first. */
    navigate: (page: Page) => Ref.set(pendingPage, Option.some(page)).pipe(Effect.zipRight(PubSub.publish(actions, { _tag: "Navigate", page }))),
    openModel: (action: OpenModel) => Ref.set(pendingAddModel, Option.some(action)).pipe(Effect.zipRight(PubSub.publish(actions, action))),
    /** Holds `page` for the next renderer without sending it, for a window that has none yet. */
    request: (page: Page) => Ref.set(pendingPage, Option.some(page)),
    subscribe: Stream.unwrapScoped(Effect.gen(function* () {
      const subscription = yield* PubSub.subscribe(actions)
      const page = yield* Ref.getAndSet(pendingPage, Option.none())
      const model = yield* Ref.getAndSet(pendingAddModel, Option.none())
      const pending: ReadonlyArray<Action> = [
        ...Option.toArray(Option.map(page, (value): Action => ({ _tag: "Navigate", page: value }))),
        ...Option.toArray(model),
      ]
      return Stream.concat(Stream.fromIterable(pending), Stream.fromQueue(subscription).pipe(Stream.tap(received)))
    })),
  }
})
