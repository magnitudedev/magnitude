import { describe, expect, test } from 'vitest'
import { hasCloudUsageAuth } from './usage-auth'

describe('cloud usage auth', () => {
  test('accepts configured and environment API keys', () => {
    expect(hasCloudUsageAuth(true, { source: 'none' })).toBe(true)
    expect(hasCloudUsageAuth(false, { source: 'env', key: 'key', envVarName: 'MAGNITUDE_API_KEY' })).toBe(true)
  })

  test('does not query cloud usage without a key', () => {
    expect(hasCloudUsageAuth(false, { source: 'none' })).toBe(false)
  })
})
