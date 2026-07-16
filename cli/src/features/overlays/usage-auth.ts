import type { AuthSource } from '../../state/cli-atoms'

export function hasCloudUsageAuth(keyAlreadySet: boolean, authSource: AuthSource): boolean {
  return keyAlreadySet
    || authSource.source === 'env'
}
