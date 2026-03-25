import { describe, expect, test } from 'bun:test'
import { createStackMachine, type Op } from '../machine'

describe('createStackMachine', () => {
  test('push/pop/peek basics', () => {
    const events: string[] = []
    const machine = createStackMachine<string, string>('root', (event) => events.push(event))

    machine.apply([{ type: 'push', frame: 'a' }])
    expect(machine.peek()).toBe('a')
    expect(machine.stack).toEqual(['root', 'a'])

    machine.apply([{ type: 'pop' }])
    expect(machine.peek()).toBe('root')
    expect(machine.stack).toEqual(['root'])
    expect(events).toEqual([])
  })

  test('popUntil stops at predicate match (exclusive)', () => {
    const machine = createStackMachine<string, string>('root', () => {})
    machine.apply([
      { type: 'push', frame: 'a' },
      { type: 'push', frame: 'b' },
      { type: 'push', frame: 'c' },
      { type: 'popUntil', predicate: (frame) => frame === 'a' },
    ])

    expect(machine.stack).toEqual(['root', 'a'])
    expect(machine.peek()).toBe('a')
  })

  test('replace swaps top', () => {
    const machine = createStackMachine<string, string>('root', () => {})
    machine.apply([{ type: 'replace', frame: 'new-root' }])
    expect(machine.stack).toEqual(['new-root'])

    machine.apply([{ type: 'push', frame: 'a' }])
    machine.apply([{ type: 'replace', frame: 'b' }])
    expect(machine.stack).toEqual(['new-root', 'b'])
  })

  test('done stops current batch and future batches', () => {
    const events: string[] = []
    const machine = createStackMachine<string, string>('root', (event) => events.push(event))

    machine.apply([
      { type: 'emit', event: 'before' },
      { type: 'done' },
      { type: 'emit', event: 'after' },
      { type: 'push', frame: 'x' },
    ])

    expect(machine.done).toBe(true)
    expect(events).toEqual(['before'])
    expect(machine.stack).toEqual(['root'])

    machine.apply([
      { type: 'emit', event: 'future' },
      { type: 'push', frame: 'y' },
    ])

    expect(events).toEqual(['before'])
    expect(machine.stack).toEqual(['root'])
  })

  test('pop never removes initial frame', () => {
    const machine = createStackMachine<string, string>('root', () => {})
    machine.apply([{ type: 'pop' }, { type: 'pop' }])
    expect(machine.stack).toEqual(['root'])
    expect(machine.peek()).toBe('root')
  })

  test('empty ops array is no-op', () => {
    const machine = createStackMachine<string, string>('root', () => {})
    machine.apply([])
    expect(machine.stack).toEqual(['root'])
    expect(machine.done).toBe(false)
  })

  test('replace on stack with only initial frame works', () => {
    const machine = createStackMachine<string, string>('root', () => {})
    machine.apply([{ type: 'replace', frame: 'updated-root' }])
    expect(machine.stack).toEqual(['updated-root'])
  })

  test('multiple ops in one apply call', () => {
    const events: string[] = []
    const machine = createStackMachine<string, string>('root', (event) => events.push(event))

    const ops: Op<string, string>[] = [
      { type: 'push', frame: 'a' },
      { type: 'emit', event: 'one' },
      { type: 'replace', frame: 'b' },
      { type: 'pop' },
      { type: 'emit', event: 'two' },
    ]

    machine.apply(ops)

    expect(machine.stack).toEqual(['root'])
    expect(machine.peek()).toBe('root')
    expect(events).toEqual(['one', 'two'])
  })
})
