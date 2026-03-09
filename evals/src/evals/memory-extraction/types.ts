import type { Scenario, Check, ChatMessage } from '../../types'

export type MemoryCategory = 'codebase' | 'workflow'
export type MemoryScenarioGroup = 'decision' | 'quality' | 'multi'

export interface MemoryAddition {
  category: MemoryCategory
  content: string
  evidence?: string
}

export interface MemoryUpdate {
  existing: string
  replacement: string
  evidence?: string
}

export interface MemoryDeletion {
  existing: string
  evidence?: string
}

export interface MemoryDiffResult {
  reasoning: string
  additions: MemoryAddition[]
  updates: MemoryUpdate[]
  deletions: MemoryDeletion[]
}

export interface JudgeCheck {
  id: string
  description: string
  question: string
}

export interface MemoryExpectedChecks {
  expectEmpty?: boolean
  minTotalOps?: number
  maxTotalOps?: number
  requiredAdditionCategories?: MemoryCategory[]
  allowedAdditionCategories?: MemoryCategory[]
  expectUpdateOrDeletion?: boolean
  forbidDuplicateOfExisting?: boolean
}

export interface MemorySessionCase {
  transcript: string
  currentMemory: string
  expected?: MemoryExpectedChecks
}

export interface MemorySingleScenario extends Scenario {
  group: 'decision' | 'quality'
  transcript: string
  currentMemory: string
  expected?: MemoryExpectedChecks
  judgeChecks?: JudgeCheck[]
}

export interface MemoryMultiScenario extends Scenario {
  group: 'multi'
  sessions: MemorySessionCase[]
  judgeChecks?: JudgeCheck[]
}

export type MemoryEvalScenario = MemorySingleScenario | MemoryMultiScenario

export function makeBaseScenario(
  id: string,
  description: string,
  messages: ChatMessage[] = [],
  checks: Check[] = []
): Scenario {
  return { id, description, messages, checks }
}