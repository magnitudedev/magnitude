export interface ThinkingLens {
  name: string
  trigger: string
  description: string
}

export function defineThinkingLens(lens: ThinkingLens): ThinkingLens {
  return lens
}