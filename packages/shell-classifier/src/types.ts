export type ShellSafetyTier = 'readonly' | 'normal' | 'mass-destructive' | 'forbidden'

export type ClassificationResult = {
  tier: ShellSafetyTier
  reason: string | null
}