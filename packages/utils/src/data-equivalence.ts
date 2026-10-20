import { isDeepStrictEqual } from "node:util"
import type { Equivalence } from "effect"

/**
 * Value equality of immutable decoded data: same prototypes, same own fields, recursively, so
 * Options, tagged unions and Schema classes compare by content. Use it where a large state is
 * compared on every refresh; `Schema.equivalence` rebuilds a parser for each union member it meets,
 * which blocks the event loop for whole seconds on lists of models.
 */
export const dataEquivalence = <A>(): Equivalence.Equivalence<A> => isDeepStrictEqual
