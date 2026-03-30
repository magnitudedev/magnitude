import { describe, expect, test } from 'bun:test'
import { Data } from 'effect'
import { defineFSM, type StateUnion } from '../define'

class Idle extends Data.TaggedClass('idle')<{
  readonly shared: string
}> {}

class Active extends Data.TaggedClass('active')<{
  readonly shared: string
  readonly count: number
}> {}

class Done extends Data.TaggedClass('done')<{
  readonly shared: string
  readonly count: number
  readonly reason: string
}> {}

const machine = defineFSM(
  { idle: Idle, active: Active, done: Done },
  { idle: ['active'], active: ['idle', 'done'], done: [] } as const
)

describe('defineFSM', () => {
  test('valid transitions work and return target instances', () => {
    const idle = new Idle({ shared: 'x' })
    const active = machine.transition(idle, 'active', { count: 1 })
    const done = machine.transition(active, 'done', { reason: 'ok' })

    expect(active._tag).toBe('active')
    expect(active.count).toBe(1)
    expect(done._tag).toBe('done')
    expect(done.reason).toBe('ok')
    expect(done.count).toBe(1)
  })

  test('invalid transitions throw at runtime', () => {
    const idle = new Idle({ shared: 'x' })
    // @ts-expect-error — intentionally testing invalid transition at runtime
    expect(() => machine.transition(idle, 'done', { count: 1, reason: 'no' })).toThrow(
      'Invalid FSM transition'
    )
  })

  test('TransitionUpdates requires new fields while shared remain optional', () => {
    const idle = new Idle({ shared: 'x' })
    const active = machine.transition(idle, 'active', { count: 2 })
    const done = machine.transition(active, 'done', { reason: 'because' })

    expect(active.shared).toBe('x')
    expect(done.count).toBe(2)
    expect(done.reason).toBe('because')
  })

  test('hold preserves type and updates fields', () => {
    const active = new Active({ shared: 'x', count: 1 })
    const updated = machine.hold(active, { count: 2 })

    expect(updated._tag).toBe('active')
    expect(updated.count).toBe(2)
  })

  test('match returns values per state', () => {
    const active = new Active({ shared: 'x', count: 3 })
    const result = machine.match(active, {
      idle: () => 'idle',
      active: (s) => `active:${s.count}`,
      done: (s) => `done:${s.reason}`
    })

    expect(result).toBe('active:3')
  })

  test('is narrows correctly', () => {
    const state = new Active({
      shared: 'x',
      count: 4
    }) as StateUnion<typeof machine.stateClasses>

    if (machine.is(state, 'active')) {
      expect(state.count).toBe(4)
    }
    if (machine.is(state, 'idle')) {
      expect(state.shared).toBe('x')
    }
  })

  test('canTransition, isTerminal, getTerminalStates behave correctly', () => {
    expect(machine.canTransition('idle', 'active')).toBe(true)
    expect(machine.canTransition('idle', 'done')).toBe(false)
    expect(machine.isTerminal('done')).toBe(true)
    expect(machine.isTerminal('active')).toBe(false)
    expect(machine.getTerminalStates()).toEqual(['done'])
  })
})

// Compile-time checks
if (false) {
  const idle = new Idle({ shared: 'x' })
  const active = machine.transition(idle, 'active', { count: 1 })

  // @ts-expect-error forbidden transition
  machine.transition(idle, 'done', { count: 1, reason: 'nope' })

  // @ts-expect-error missing required field for idle -> active
  machine.transition(idle, 'active', {})

  // shared fields optional, new fields required (reason required for active -> done)
  // @ts-expect-error missing required reason
  machine.transition(active, 'done', {})
}
