import { afterEach, describe, expect, test } from 'bun:test'
import { resolveTerminalWidth } from './use-terminal-width'

const originalColumnsDescriptor = Object.getOwnPropertyDescriptor(process.stdout, 'columns')

function setStdoutColumns(value: number | undefined) {
  Object.defineProperty(process.stdout, 'columns', {
    value,
    configurable: true,
    writable: true,
  })
}

afterEach(() => {
  if (originalColumnsDescriptor) {
    Object.defineProperty(process.stdout, 'columns', originalColumnsDescriptor)
  }
})

describe('resolveTerminalWidth', () => {
  test('prefers renderer terminal width', () => {
    setStdoutColumns(120)
    expect(resolveTerminalWidth({ terminal: { width: 42 } })).toBe(42)
  })

  test('falls back to renderer screen width', () => {
    setStdoutColumns(120)
    expect(resolveTerminalWidth({ screen: { width: 55 } })).toBe(55)
  })

  test('falls back to process stdout columns', () => {
    setStdoutColumns(77)
    expect(resolveTerminalWidth({})).toBe(77)
  })

  test('uses 80 default when no width is available', () => {
    setStdoutColumns(undefined)
    expect(resolveTerminalWidth({})).toBe(80)
  })

  test('never returns less than 1', () => {
    setStdoutColumns(0)
    expect(resolveTerminalWidth({ terminal: { width: 0 } })).toBe(1)
  })
})