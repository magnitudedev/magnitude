import type { ApiKeyState } from '@magnitudedev/client-common'
import type { AuthSource } from '../../state/cli-atoms'

type AuthAction = (key: string) => Promise<void>
type ClearAuthAction = () => Promise<void>

/** Auth display info for the settings overlay. */
export type AuthInfo =
  | {
    source: 'config'
    key: null
    maskedKey: string | null
    envVarName: null
    save: AuthAction
    clear: ClearAuthAction
  }
  | {
    source: 'env' | 'env-local'
    key: string
    maskedKey: null
    envVarName: string
    save: AuthAction
    clear: ClearAuthAction
  }
  | {
    source: 'none'
    key: null
    maskedKey: null
    envVarName: null
    save: AuthAction
    clear: ClearAuthAction
  }

export function deriveSettingsAuthInfo({
  apiKey,
  authSource,
  save,
  clear,
}: {
  apiKey: ApiKeyState
  authSource: AuthSource
  save: AuthAction
  clear: ClearAuthAction
}): AuthInfo {
  if (authSource.source === 'env-local') {
    return {
      source: 'env-local',
      key: authSource.key,
      maskedKey: null,
      envVarName: authSource.envVarName,
      save,
      clear,
    }
  }

  if (apiKey.status === 'config') {
    return {
      source: 'config',
      key: null,
      maskedKey: apiKey.maskedKey ?? null,
      envVarName: null,
      save,
      clear,
    }
  }

  if (authSource.source === 'env') {
    return {
      source: 'env',
      key: authSource.key,
      maskedKey: null,
      envVarName: authSource.envVarName,
      save,
      clear,
    }
  }

  return {
    source: 'none',
    key: null,
    maskedKey: null,
    envVarName: null,
    save,
    clear,
  }
}
