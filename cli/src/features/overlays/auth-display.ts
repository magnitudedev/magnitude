import type { ApiKeyState } from '@magnitudedev/client-common'
import type { AuthSource } from '../../state/cli-atoms'

type AuthAction = (key: string) => void
type ClearAuthAction = () => void

/** Auth display info for the settings overlay. */
interface AuthActions {
  readonly save: AuthAction
  readonly clear: ClearAuthAction
  readonly saving: boolean
  readonly error: string | null
}

export type AuthInfo = AuthActions & (
  | {
    source: 'config'
    key: null
    maskedKey: string | null
    envVarName: null
  }
  | {
    source: 'env' | 'env-local'
    key: string
    maskedKey: null
    envVarName: string
  }
  | {
    source: 'none'
    key: null
    maskedKey: null
    envVarName: null
  })

export function deriveSettingsAuthInfo({
  apiKey,
  authSource,
  save,
  clear,
  saving = false,
  error = null,
}: {
  apiKey: ApiKeyState
  authSource: AuthSource
  save: AuthAction
  clear: ClearAuthAction
  saving?: boolean
  error?: string | null
}): AuthInfo {
  const actions = { save, clear, saving, error }
  if (authSource.source === 'env-local') {
    return {
      source: 'env-local',
      key: authSource.key,
      maskedKey: null,
      envVarName: authSource.envVarName,
      ...actions,
    }
  }

  if (apiKey.status === 'config') {
    return {
      source: 'config',
      key: null,
      maskedKey: apiKey.maskedKey ?? null,
      envVarName: null,
      ...actions,
    }
  }

  if (authSource.source === 'env') {
    return {
      source: 'env',
      key: authSource.key,
      maskedKey: null,
      envVarName: authSource.envVarName,
      ...actions,
    }
  }

  return {
    source: 'none',
    key: null,
    maskedKey: null,
    envVarName: null,
    ...actions,
  }
}
