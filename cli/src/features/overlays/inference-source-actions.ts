export const INFERENCE_SOURCE_ACTIONS = {
  local: {
    key: 'l',
    label: 'Manage local models',
  },
} as const

export type InferenceSourceAction = keyof typeof INFERENCE_SOURCE_ACTIONS

export function getInferenceSourceAction(keyName: string): InferenceSourceAction | null {
  if (keyName === INFERENCE_SOURCE_ACTIONS.local.key) return 'local'
  return null
}
