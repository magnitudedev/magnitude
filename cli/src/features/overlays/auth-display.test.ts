import { describe, expect, test } from 'vitest'
import type { ApiKeyState } from '@magnitudedev/client-common'
import type { AuthSource } from '../../state/cli-atoms'
import { deriveSettingsAuthInfo } from './auth-display'

const save = async () => {}
const clear = async () => {}

function derive(apiKey: ApiKeyState, authSource: AuthSource) {
  return deriveSettingsAuthInfo({ apiKey, authSource, save, clear })
}

describe('deriveSettingsAuthInfo', () => {
  test('uses environment auth ahead of configured storage', () => {
    const auth = derive(
      { status: 'config', maskedKey: 'mg_sk_conf………2222' },
      { source: 'env', key: 'env-key', envVarName: 'MAGNITUDE_API_KEY' },
    )

    expect(auth).toMatchObject({
      source: 'env',
      key: 'env-key',
      maskedKey: null,
      envVarName: 'MAGNITUDE_API_KEY',
    })
  })

  test('uses configured storage when no env auth exists', () => {
    const auth = derive(
      { status: 'config', maskedKey: 'mg_sk_conf………3333' },
      { source: 'none' },
    )

    expect(auth).toMatchObject({
      source: 'config',
      key: null,
      maskedKey: 'mg_sk_conf………3333',
      envVarName: null,
    })
  })

  test('uses normal env auth when no configured key exists', () => {
    const auth = derive(
      { status: 'none' },
      { source: 'env', key: 'env-key', envVarName: 'MAGNITUDE_API_KEY' },
    )

    expect(auth).toMatchObject({
      source: 'env',
      key: 'env-key',
      maskedKey: null,
      envVarName: 'MAGNITUDE_API_KEY',
    })
  })

  test('uses none when no configured or env key exists', () => {
    const auth = derive(
      { status: 'none' },
      { source: 'none' },
    )

    expect(auth).toMatchObject({
      source: 'none',
      key: null,
      maskedKey: null,
      envVarName: null,
    })
  })
})
