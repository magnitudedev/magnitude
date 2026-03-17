export type Format = 'xml-act' | 'xml-v2' | 'declare'

export interface Scenario {
  id: string
  description: string
  userMessage: string
  shouldContinue: boolean
  shouldUsePd: boolean | null    // null = not applicable
  checkNoHallucination: boolean | null  // null = not applicable
  checkEscaping?: string  // if set, raw output must contain this exact literal string
}

export interface Scores {
  syntax_valid: boolean
  single_turn: boolean
  turn_control_correct: boolean
  pd_used: boolean | null           // null = not scored for this scenario
  no_hallucinated_results: boolean | null  // null = not scored for this scenario
  escaping_correct: boolean | null  // null = not scored for this scenario
}

export interface ModelSpec {
  provider: string
  model: string
  label: string
}

export interface Result {
  model: string
  format: Format
  scenario: string
  raw_output: string
  scores: Scores
}

export interface RunOutput {
  timestamp: string
  results: Result[]
}