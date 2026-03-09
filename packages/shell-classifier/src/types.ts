export type ShellSafetyTier = 'readonly' | 'normal' | 'forbidden'

export type ClassificationResult = {
  tier: ShellSafetyTier
  reason: string | null
}