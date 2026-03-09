/**
 * FSM.define - Type-safe Finite State Machine
 *
 * Uses Effect's Data.TaggedClass for state classes. FSM provides typed
 * transition() and hold() functions.
 *
 * @example
 * ```typescript
 * import { Data } from 'effect'
 * import { FSM } from 'sage-core'
 *
 * class Pending extends Data.TaggedClass('pending')<{ id: string; content: string }> {}
 * class Streaming extends Data.TaggedClass('streaming')<{ id: string; content: string }> {}
 * class Completed extends Data.TaggedClass('completed')<{ id: string; content: string }> {}
 *
 * const ResponseFSM = FSM.define({
 *   transitions: {
 *     pending: ['streaming'],
 *     streaming: ['completed'],
 *     completed: []
 *   } as const,
 *   states: [Pending, Streaming, Completed]
 * })
 *
 * const p = new Pending({ id: '1', content: '' })
 * const s = ResponseFSM.transition(p, 'streaming', { content: 'hello' })
 * const s2 = ResponseFSM.hold(s, { content: s.content + ' world' })
 * ```
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * FSM transition matrix type.
 */
export type TransitionMatrix = Record<string, readonly string[]>

/**
 * Extract all state names from a transition matrix.
 */
export type StateNames<T extends TransitionMatrix> = keyof T & string

/**
 * Extract valid target states from a given source state.
 */
export type ValidTargets<T extends TransitionMatrix, From extends StateNames<T>> =
  T[From][number] & string

/**
 * Base interface for tagged items.
 */
export interface TaggedItem<Tag extends string = string> {
  readonly _tag: Tag
}

/**
 * Extract the _tag literal type from a state class.
 */
export type TagOf<C> = C extends new (...args: never[]) => { readonly _tag: infer T extends string } ? T : never

/**
 * Extract props type from a state class (excluding _tag).
 */
export type PropsOf<C> = C extends new (props: infer P) => unknown ? P : never

/**
 * Build a record mapping tags to state classes from an array of classes.
 */
export type StateClassRecord<Classes extends readonly AnyStateClass[]> = {
  [C in Classes[number] as TagOf<C>]: C
}

/**
 * Get the instance type of a state class.
 */
export type InstanceOf<C> = C extends new (...args: never[]) => infer I ? I : never

/**
 * Union of all state instances from an array of classes.
 */
export type StateUnion<Classes extends readonly AnyStateClass[]> =
  InstanceOf<Classes[number]>

/**
 * Any state class (for constraints).
 * Uses `any` for props like Effect's Data.TaggedClass does.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyStateClass = new (props: any) => TaggedItem

/**
 * Compute required updates when transitioning from one state to another.
 * - Fields new to target (not on source): required
 * - Fields shared with source: optional (inherited from source instance)
 * - 'id' and '_tag' excluded (handled by framework)
 */
export type TransitionUpdates<From, ToProps> =
  & Omit<ToProps, keyof From | '_tag'>
  & Partial<Pick<ToProps, Extract<keyof From, keyof ToProps>>>

// ---------------------------------------------------------------------------
// FSM Instance Type
// ---------------------------------------------------------------------------

export interface FSMInstance<
  T extends TransitionMatrix,
  Classes extends readonly AnyStateClass[]
> {
  /**
   * The transition matrix.
   */
  readonly transitions: T

  /**
   * Array of all state names.
   */
  readonly states: StateNames<T>[]

  /**
   * Record mapping tags to state classes.
   */
  readonly stateClasses: StateClassRecord<Classes>

  /**
   * Transition to a new state. Validates the transition is allowed.
   *
   * Fields new to target state are required; fields shared with source are optional.
   */
  transition<
    From extends StateUnion<Classes>,
    To extends StateNames<T> & keyof StateClassRecord<Classes>
  >(
    instance: From,
    target: To,
    updates: TransitionUpdates<From, PropsOf<StateClassRecord<Classes>[To]>>
  ): InstanceOf<StateClassRecord<Classes>[To]>

  /**
   * Stay in current state with updated data.
   */
  hold<From extends StateUnion<Classes>>(
    instance: From,
    updates: Partial<PropsOf<Classes[number]>>
  ): From

  /**
   * Check if a transition is valid.
   */
  canTransition(from: StateNames<T>, to: StateNames<T>): boolean

  /**
   * Get valid target states from a source state.
   */
  getValidTargets(from: StateNames<T>): readonly string[]

  /**
   * Check if a state is terminal.
   */
  isTerminal(state: StateNames<T>): boolean

  /**
   * Get all terminal states.
   */
  getTerminalStates(): StateNames<T>[]
}

// ---------------------------------------------------------------------------
// FSM Define
// ---------------------------------------------------------------------------

/**
 * Define a type-safe FSM.
 */
export function define<
  const T extends TransitionMatrix,
  const Classes extends readonly AnyStateClass[]
>(config: {
  transitions: T
  states: Classes
}): FSMInstance<T, Classes> {
  type StateName = StateNames<T>

  // Build tag -> class mapping
  const classRecord: Record<string, AnyStateClass> = {}
  for (const StateClass of config.states) {
    // Create a dummy instance to get the tag
    try {
      const dummy = new StateClass({})
      classRecord[dummy._tag] = StateClass
    } catch {
      throw new Error(`Could not determine _tag for state class: ${StateClass.name}`)
    }
  }

  const stateNames = Object.keys(config.transitions) as StateName[]

  return {
    transitions: config.transitions,
    states: stateNames,
    stateClasses: classRecord as StateClassRecord<Classes>,

    transition(instance, target, updates) {
      const fromTag = instance._tag
      const validTargets = config.transitions[fromTag]

      if (!validTargets?.includes(target)) {
        throw new Error(
          `Invalid FSM transition: ${fromTag} -> ${target}\n` +
          `Valid targets from ${fromTag}: ${validTargets?.join(', ') || 'none'}`
        )
      }

      const TargetClass = classRecord[target]
      if (!TargetClass) {
        throw new Error(`No state class registered for tag: ${target}`)
      }

      return new TargetClass({ ...(instance as Record<string, unknown>), ...updates }) as InstanceOf<StateClassRecord<Classes>[typeof target]>
    },

    hold(instance, updates) {
      const CurrentClass = classRecord[instance._tag]
      if (!CurrentClass) {
        throw new Error(`No state class registered for tag: ${instance._tag}`)
      }
      return new CurrentClass({ ...instance, ...updates }) as typeof instance
    },

    canTransition(from: StateName, to: StateName): boolean {
      return config.transitions[from]?.includes(to) ?? false
    },

    getValidTargets(from: StateName): readonly string[] {
      return config.transitions[from] ?? []
    },

    isTerminal(state: StateName): boolean {
      const targets = config.transitions[state]
      return !targets || targets.length === 0
    },

    getTerminalStates(): StateName[] {
      return stateNames.filter(state => {
        const targets = config.transitions[state]
        return !targets || targets.length === 0
      })
    }
  }
}
