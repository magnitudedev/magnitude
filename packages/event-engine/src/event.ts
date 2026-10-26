import { Data } from "effect"

export const EventAny: unique symbol = Symbol.for("@magnitudedev/event-engine/Event.any")

export const Event = {
  any: EventAny,
} as const

export interface EventRefinement<Event, Selected extends Event> {
  readonly name: string
  readonly refinement: (event: Event) => event is Selected
}

type LiteralPropertyKey<Value> = [Value] extends [PropertyKey]
  ? string extends Value ? never
    : number extends Value ? never
      : symbol extends Value ? never
        : Value
  : never

export type DiscriminatorKey<Event> = {
  [Key in keyof Event]-?: [LiteralPropertyKey<Event[Key]>] extends [never] ? never : Key
}[keyof Event] & PropertyKey

export type DiscriminatorValue<Event, Key extends PropertyKey | undefined> =
  Key extends keyof Event ? Event[Key] & PropertyKey : never

export type EventVariant<
  Event,
  Key extends PropertyKey | undefined,
  Value extends PropertyKey,
> = Event extends unknown
  ? Key extends keyof Event
    ? Event[Key] extends Value ? Event : never
    : never
  : never

export class DiscriminatorRequired extends Data.TaggedError("DiscriminatorRequired")<{}> {}

export interface EventMatcher<Event> {
  readonly name: string
  readonly matches: (event: Event) => boolean
}

export type EventSelection<Event, Key extends DiscriminatorKey<Event> | undefined> = {
  readonly discriminator: Key
  readonly matcher: <Selected extends Event>(
    selector: PropertyKey | typeof EventAny | EventRefinement<Event, Selected>,
  ) => EventMatcher<Event>
}

export const makeEventSelection = <
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
>(discriminator: Key): EventSelection<Event, Key> => ({
  discriminator,
  matcher: (selector) => {
    if (selector === EventAny) {
      return { name: "*", matches: () => true }
    }
    if (typeof selector === "object" && selector !== null && "refinement" in selector) {
      return { name: selector.name, matches: selector.refinement }
    }
    if (discriminator === undefined) {
      throw new DiscriminatorRequired()
    }
    return {
      name: String(selector),
      matches: (event) => (event as Record<PropertyKey, unknown>)[discriminator] === selector,
    }
  },
})
