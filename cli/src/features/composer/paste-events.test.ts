import { describe, expect, test } from 'bun:test'
import { decodeNativePasteText } from './paste-events'

describe('decodeNativePasteText', () => {
  test('decodes native paste bytes', () => {
    const text = decodeNativePasteText({ bytes: new Uint8Array([65, 66]) })
    expect(text).toBe('AB')
  })
})
