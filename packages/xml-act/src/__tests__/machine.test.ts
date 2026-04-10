import { describe, expect, test } from 'vitest'
import { createStackMachine, type Op } from '../machine'

type Frame = { readonly type: string }

const frame = (type: string): Frame => ({ type })

describe('createStackMachine', () => {
  test('push/pop/peek basics', () => {
    const events: string[] = []
    const machine = createStackMachine<Frame, string>(frame('root'), (event) => events.push(event))

    machine.apply([{ type: 'push', frame: frame('a') }])
    expect(machine.peek()).toEqual(frame('a'))
    expect(machine.stack).toEqual([frame('root'), frame('a')])

    machine.apply([{ type: 'pop' }])
    expect(machine.peek()).toEqual(frame('root'))
    expect(machine.stack).toEqual([frame('root')])
    expect(events).toEqual([])
  })

  test('popUntil stops at predicate match (exclusive)', () => {
    const machine = createStackMachine<Frame, string>(frame('root'), () => {})
    machine.apply([
      { type: 'push', frame: frame('a') },
      { type: 'push', frame: frame('b') },
      { type: 'push', frame: frame('c') },
      { type: 'popUntil', predicate: (f) => f.type === 'a' },
    ])

    expect(machine.stack).toEqual([frame('root'), frame('a')])
    expect(machine.peek()).toEqual(frame('a'))
  })

  test('replace swaps top', () => {
    const machine = createStackMachine<Frame, string>(frame('root'), () => {})
    machine.apply([{ type: 'replace', frame: frame('new-root') }])
    expect(machine.stack).toEqual([frame('new-root')])

    machine.apply([{ type: 'push', frame: frame('a') }])
    machine.apply([{ type: 'replace', frame: frame('b') }])
    expect(machine.stack).toEqual([frame('new-root'), frame('b')])
  })

  test('done stops current batch and future batches', () => {
    const events: string[] = []
    const machine = createStackMachine<Frame, string>(frame('root'), (event) => events.push(event))

    machine.apply([
      { type: 'emit', event: 'before' },
      { type: 'done' },
      { type: 'emit', event: 'after' },
      { type: 'push', frame: frame('x') },
    ])

    expect(machine.mode).toBe('done')
    expect(events).toEqual(['before'])
    expect(machine.stack).toEqual([frame('root')])

    machine.apply([
      { type: 'emit', event: 'future' },
      { type: 'push', frame: frame('y') },
    ])

    expect(events).toEqual(['before'])
    expect(machine.stack).toEqual([frame('root')])
  })

  test('pop never removes initial frame', () => {
    const machine = createStackMachine<Frame, string>(frame('root'), () => {})
    machine.apply([{ type: 'pop' }, { type: 'pop' }])
    expect(machine.stack).toEqual([frame('root')])
    expect(machine.peek()).toEqual(frame('root'))
  })

  test('empty ops array is no-op', () => {
    const machine = createStackMachine<Frame, string>(frame('root'), () => {})
    machine.apply([])
    expect(machine.stack).toEqual([frame('root')])
    expect(machine.mode).toBe('active')
  })

  test('replace on stack with only initial frame works', () => {
    const machine = createStackMachine<Frame, string>(frame('root'), () => {})
    machine.apply([{ type: 'replace', frame: frame('updated-root') }])
    expect(machine.stack).toEqual([frame('updated-root')])
  })

  test('multiple ops in one apply call', () => {
    const events: string[] = []
    const machine = createStackMachine<Frame, string>(frame('root'), (event) => events.push(event))

    const ops: Op<Frame, string>[] = [
      { type: 'push', frame: frame('a') },
      { type: 'emit', event: 'one' },
      { type: 'replace', frame: frame('b') },
      { type: 'pop' },
      { type: 'emit', event: 'two' },
    ]

    machine.apply(ops)

    expect(machine.stack).toEqual([frame('root')])
    expect(machine.peek()).toEqual(frame('root'))
    expect(events).toEqual(['one', 'two'])
  })
})
