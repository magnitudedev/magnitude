import { expect, test } from 'bun:test'
import { capitalize, slugify, titleCase } from '../src/lib'

test('capitalize makes first letter uppercase', () => {
  expect(capitalize('hello')).toBe('Hello')
})

test('capitalize handles empty string', () => {
  expect(capitalize('')).toBe('')
})

test('capitalize handles already capitalized', () => {
  expect(capitalize('Hello')).toBe('Hello')
})

test('slugify converts to url-safe string', () => {
  expect(slugify('Hello World!')).toBe('hello-world')
})

test('titleCase capitalizes each word', () => {
  expect(titleCase('hello world')).toBe('Hello World')
})

test('titleCase handles single word', () => {
  expect(titleCase('hello')).toBe('Hello')
})
