const AmbientTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Ambient")

/**
 * A live value outside the log (event-core's ambient). It starts at `initial`; an update settles
 * like an event: projection ambient handlers fold the new value and the signals they emit cascade
 * and are released. Nothing is stored or replayed, so an engine starts from `initial`.
 */
export interface AmbientDefinition<Value> {
  readonly [AmbientTypeId]: typeof AmbientTypeId
  readonly name: string
  readonly initial: Value
}

/** Reads ambients' current values, synchronously, in projection handlers. */
export interface AmbientReader {
  readonly get: <Value>(ambient: AmbientDefinition<Value>) => Value
}

export const Ambient = {
  define: <Value>(config: { readonly name: string; readonly initial: Value }): AmbientDefinition<Value> => ({
    [AmbientTypeId]: AmbientTypeId,
    name: config.name,
    initial: config.initial,
  }),
}
