import { describe, expect, test } from 'vitest'
import { basename } from 'node:path'
import { getDevelopmentRgPath } from './resolve'

describe('development ripgrep path', () => {
  test('uses the platform-specific executable name', () => {
    const moduleUrl = new URL('./resolve.ts', import.meta.url).href
    expect(basename(getDevelopmentRgPath(moduleUrl, false))).toBe('rg')
    expect(basename(getDevelopmentRgPath(moduleUrl, true))).toBe('rg.exe')
  })
})
