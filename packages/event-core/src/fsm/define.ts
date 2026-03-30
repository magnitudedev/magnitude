export type AnyTaggedState = { readonly _tag: string }

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyStateClass = new (props: any) => AnyTaggedState

export type StateClassRecord = Record<string, AnyStateClass>

export type TransitionMatrix = Record<string, readonly string[]>

type TransitionMatrixForStates<TStates extends StateClassRecord> = {
  readonly [K in keyof TStates & string]: readonly (keyof TStates & string)[]
}

export type StateNames<TTransitions extends TransitionMatrix> = keyof TTransitions & string

export type InstanceOf<C> = C extends new (...args: never[]) => infer I ? I : never

export type StateUnion<TStates extends StateClassRecord> =
  InstanceOf<TStates[keyof TStates]>

export type PropsOf<C> = C extends new (props: infer P) => unknown ? P : never

export type TagOf<C> = C extends new (...args: never[]) => { readonly _tag: infer T extends string } ? T : never

export type ValidTargets<
  TTransitions extends TransitionMatrix,
  From extends StateNames<TTransitions>
> = TTransitions[From] extends readonly (infer TTo)[]
  ? TTo & string
  : never

export type TransitionUpdates<From, ToProps> =
  & Omit<ToProps, keyof From | '_tag'>
  & Partial<Pick<ToProps, Extract<keyof From, keyof ToProps>>>

type ClassForTag<
  TStates extends StateClassRecord,
  TTag extends keyof TStates & string
> = TStates[TTag]

type TagsReachableFrom<
  TTransitions extends TransitionMatrix,
  TFrom extends string
> = TFrom extends keyof TTransitions
  ? TTransitions[TFrom][number] & string
  : never

export interface FSMDefinition<
  TStates extends StateClassRecord,
  TTransitions extends TransitionMatrix
> {
  readonly states: readonly (keyof TStates & string)[]
  readonly stateClasses: TStates
  readonly transitions: TTransitions

  transition<
    From extends StateUnion<TStates>,
    FromTag extends From['_tag'] & keyof TStates & string,
    To extends TagsReachableFrom<TTransitions, FromTag> & keyof TStates & string
  >(
    from: From,
    target: To,
    updates: TransitionUpdates<From, PropsOf<ClassForTag<TStates, To>>>
  ): InstanceOf<ClassForTag<TStates, To>>

  hold<From extends StateUnion<TStates>>(
    from: From,
    updates: Partial<Omit<From, '_tag'>>
  ): From

  match<
    TState extends StateUnion<TStates>,
    THandlers extends {
      [K in keyof TStates & string]: (state: InstanceOf<TStates[K]>) => unknown
    }
  >(
    state: TState,
    handlers: THandlers
  ): ReturnType<THandlers[TState['_tag'] & keyof THandlers]>

  is<
    TState extends StateUnion<TStates>,
    TTag extends keyof TStates & string
  >(
    state: TState,
    tag: TTag
  ): state is Extract<TState, InstanceOf<TStates[TTag]>>

  canTransition(
    from: keyof TStates & string,
    to: keyof TStates & string
  ): boolean

  isTerminal(state: keyof TStates & string): boolean

  getTerminalStates(): (keyof TStates & string)[]
}

export function defineFSM<
  const TStates extends StateClassRecord,
  const TTransitions extends TransitionMatrixForStates<TStates>
>(
  states: TStates,
  transitions: TTransitions
): FSMDefinition<TStates, TTransitions> {
  const stateNames = Object.keys(states) as (keyof TStates & string)[]

  const transition = <
    From extends StateUnion<TStates>,
    FromTag extends From['_tag'] & keyof TStates & string,
    To extends TagsReachableFrom<TTransitions, FromTag> & keyof TStates & string
  >(
    from: From,
    target: To,
    updates: TransitionUpdates<From, PropsOf<ClassForTag<TStates, To>>>
  ): InstanceOf<ClassForTag<TStates, To>> => {
    const fromTag = from._tag as keyof TStates & string
    const validTargets = transitions[fromTag] ?? []

    if (!validTargets.includes(target)) {
      const allowed = validTargets.length > 0 ? validTargets.join(', ') : 'none'
      throw new Error(
        `Invalid FSM transition: "${fromTag}" -> "${target}". Allowed targets: ${allowed}`
      )
    }

    const TargetClass = states[target]
    return new TargetClass({ ...from, ...updates }) as InstanceOf<ClassForTag<TStates, To>>
  }

  const hold = <From extends StateUnion<TStates>>(
    from: From,
    updates: Partial<Omit<From, '_tag'>>
  ): From => {
    const CurrentClass = states[from._tag as keyof TStates & string]
    return new CurrentClass({ ...from, ...updates }) as From
  }

  const match = <
    TState extends StateUnion<TStates>,
    THandlers extends {
      [K in keyof TStates & string]: (state: InstanceOf<TStates[K]>) => unknown
    }
  >(
    state: TState,
    handlers: THandlers
  ): ReturnType<THandlers[TState['_tag'] & keyof THandlers]> => {
    return handlers[state._tag as keyof THandlers](state as never) as ReturnType<THandlers[TState['_tag'] & keyof THandlers]>
  }

  const is = <
    TState extends StateUnion<TStates>,
    TTag extends keyof TStates & string
  >(
    state: TState,
    tag: TTag
  ): state is Extract<TState, InstanceOf<TStates[TTag]>> => state._tag === tag

  const canTransition = (
    from: keyof TStates & string,
    to: keyof TStates & string
  ): boolean => (transitions[from] ?? []).includes(to)

  const isTerminal = (state: keyof TStates & string): boolean =>
    (transitions[state] ?? []).length === 0

  const getTerminalStates = (): (keyof TStates & string)[] =>
    stateNames.filter((state) => isTerminal(state))

  return {
    states: stateNames,
    stateClasses: states,
    transitions,
    transition,
    hold,
    match,
    is,
    canTransition,
    isTerminal,
    getTerminalStates
  }
}


