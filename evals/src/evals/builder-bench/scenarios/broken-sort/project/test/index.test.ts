import { expect, test } from 'bun:test'
import { sortNumbers } from '../src/index'

test('sorts numbers ascending', () => {
  expect(sortNumbers([3, 1, 4, 1, 5])).toEqual([1, 1, 3, 4, 5])
})

test('handles empty array', () => {
  expect(sortNumbers([])).toEqual([])
})

test('handles already sorted', () => {
  expect(sortNumbers([1, 2, 3])).toEqual([1, 2, 3])
})

test('handles single element', () => {
  expect(sortNumbers([42])).toEqual([42])
})

test('handles negative numbers', () => {
  expect(sortNumbers([3, -1, 0, -5, 2])).toEqual([-5, -1, 0, 2, 3])
})
