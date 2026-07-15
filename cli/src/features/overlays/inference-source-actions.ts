export const INFERENCE_SOURCE_ACTIONS = {
  local: {
    key: 'l',
    label: 'Manage local models',
  },
  cloud: {
    key: 'c',
    label: 'Configure Cloud fallback',
  },
} as const

export type InferenceSourceAction = keyof typeof INFERENCE_SOURCE_ACTIONS

export function getInferenceSourceAction(keyName: string): InferenceSourceAction | null {
  if (keyName === INFERENCE_SOURCE_ACTIONS.local.key) return 'local'
  if (keyName === INFERENCE_SOURCE_ACTIONS.cloud.key) return 'cloud'
  return null
}
