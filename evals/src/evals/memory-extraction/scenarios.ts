import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import path from 'path'
import type { EvalVariant } from '../../types'
import type { MemoryEvalScenario, MemorySingleScenario, MemoryMultiScenario } from './types'
import { makeBaseScenario } from './types'

export const MEMORY_TEMPLATE = `# Codebase
- 


# Workflow
- 
`

const FIXTURES_DIR = fileURLToPath(new URL('../../../fixtures/memory-extraction/', import.meta.url))
const loadTranscript = (filename: string) => readFileSync(path.join(FIXTURES_DIR, filename), 'utf8')

const mem = (...lines: string[]) => `# Codebase
- ${lines[0] ?? ''}

# Workflow
- ${lines[1] ?? ''}
`

const richMem = (codebase: string[], workflow: string[]) => {
  const fmt = (items: string[]) => (items.length ? items.map((i) => `- ${i}`).join('\n') : '- ')
  return `# Codebase\n${fmt(codebase)}\n\n# Workflow\n${fmt(workflow)}\n`
}

const DECISION_SCENARIOS: MemorySingleScenario[] = [
  {
    ...makeBaseScenario('decision/no-write-routine-implementation', 'Routine implementation with approvals only'),
    group: 'decision',
    transcript: loadTranscript('decision-no-write-routine-implementation.txt'),
    currentMemory: richMem(
      ['Use TypeScript strict mode for all new files', 'All API responses follow the { data, error, meta } envelope pattern'],
      []
    ),
    expected: { expectEmpty: true, minTotalOps: 0, maxTotalOps: 0 },
  },
  {
    ...makeBaseScenario('decision/no-write-task-local-directions', 'Task-local coding directions should not persist'),
    group: 'decision',
    transcript: loadTranscript('decision-no-write-task-local-directions.txt'),
    currentMemory: richMem(
      ['Use snake_case for database column names, camelCase for TypeScript'],
      ['Always run the test suite before reporting a task as complete']
    ),
    expected: { expectEmpty: true, maxTotalOps: 0 },
  },
  {
    ...makeBaseScenario('decision/no-write-single-contextual-correction', 'Single contextual correction should not persist'),
    group: 'decision',
    transcript: loadTranscript('decision-no-write-single-contextual-correction.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { expectEmpty: true, maxTotalOps: 0 },
  },
  {
    ...makeBaseScenario('decision/no-write-ambiguous-language', 'Ambiguous hedging language should not write memory'),
    group: 'decision',
    transcript: loadTranscript('decision-no-write-ambiguous-language.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { expectEmpty: true, maxTotalOps: 0 },
  },
  {
    ...makeBaseScenario('decision/no-write-tool-heavy-minimal-signal', 'Tool-heavy but low-signal session should stay empty'),
    group: 'decision',
    transcript: loadTranscript('decision-no-write-tool-heavy-minimal-signal.txt'),
    currentMemory: richMem(
      ['Never import from barrel files in src/internal/ — causes circular deps', 'Run database migrations with prisma migrate before testing'],
      ['Use an explorer agent to map unfamiliar code areas before implementing changes']
    ),
    expected: { expectEmpty: true, maxTotalOps: 0 },
  },
  {
    ...makeBaseScenario('decision/no-write-already-in-memory', 'Already captured preference should not duplicate'),
    group: 'decision',
    transcript: loadTranscript('decision-no-write-already-in-memory.txt'),
    currentMemory: mem('Use named exports only in this repo', ''),
    expected: { expectEmpty: true, maxTotalOps: 0, forbidDuplicateOfExisting: true },
  },
  {
    ...makeBaseScenario('decision/no-write-questions-not-preferences', 'Questions should not be extracted as preferences'),
    group: 'decision',
    transcript: loadTranscript('decision-no-write-questions-not-preferences.txt'),
    currentMemory: richMem(
      ['Use TypeScript strict mode for all new files'],
      ['Create a plan artifact before starting multi-file refactors']
    ),
    expected: { expectEmpty: true, maxTotalOps: 0 },
  },
  {
    ...makeBaseScenario('decision/write-codebase-convention', 'Named-export convention should be extracted'),
    group: 'decision',
    transcript: loadTranscript('decision-write-codebase-convention.txt'),
    currentMemory: richMem(
      [],
      ['Always run the test suite before reporting a task as complete']
    ),
    expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['codebase'], allowedAdditionCategories: ['codebase'] },
  },
  {
    ...makeBaseScenario('decision/write-codebase-dependency-choice', 'Dependency/tool choice should be extracted'),
    group: 'decision',
    transcript: loadTranscript('decision-write-codebase-dependency-choice.txt'),
    currentMemory: richMem(
      ['Use TypeScript strict mode for all new files', 'All API responses follow the { data, error, meta } envelope pattern'],
      []
    ),
    expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['codebase'], allowedAdditionCategories: ['codebase'] },
  },
  {
    ...makeBaseScenario('decision/write-codebase-gotcha', 'Cross-file gotcha should be extracted'),
    group: 'decision',
    transcript: loadTranscript('decision-write-codebase-gotcha.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['codebase'], allowedAdditionCategories: ['codebase'] },
  },
  {
    ...makeBaseScenario('decision/write-communication-autonomy', 'Autonomy preference should be communication memory'),
    group: 'decision',
    transcript: loadTranscript('decision-write-communication-autonomy.txt'),
    currentMemory: richMem(
      ['Use snake_case for database column names, camelCase for TypeScript', 'Never import from barrel files in src/internal/ — causes circular deps'],
      ['Create a plan artifact before starting multi-file refactors']
    ),
    expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['workflow'], allowedAdditionCategories: ['workflow'] },
  },
  {
    ...makeBaseScenario('decision/write-communication-verbosity', 'Verbosity preference should be communication memory'),
    group: 'decision',
    transcript: loadTranscript('decision-write-communication-verbosity.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['workflow'], allowedAdditionCategories: ['workflow'] },
  },
  {
    ...makeBaseScenario('decision/write-workflow-subagent-preference', 'Subagent sequencing preference should be workflow'),
    group: 'decision',
    transcript: loadTranscript('decision-write-workflow-subagent-preference.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['workflow'], allowedAdditionCategories: ['workflow'] },
  },
  {
    ...makeBaseScenario('decision/write-workflow-verification', 'Always run tests preference should be workflow'),
    group: 'decision',
    transcript: loadTranscript('decision-write-workflow-verification.txt'),
    currentMemory: richMem(
      ['Run database migrations with prisma migrate before testing'],
      []
    ),
    expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['workflow'], allowedAdditionCategories: ['workflow'] },
  },
  {
    ...makeBaseScenario('decision/boundary-communication-vs-workflow', 'Boundary case should split communication/workflow correctly'),
    group: 'decision',
    transcript: loadTranscript('decision-boundary-communication-vs-workflow.txt'),
    currentMemory: richMem(
      ['All API responses follow the { data, error, meta } envelope pattern'],
      []
    ),
    expected: {
      minTotalOps: 2,
      maxTotalOps: 3,
      requiredAdditionCategories: ['workflow'],
      allowedAdditionCategories: ['workflow'],
    },
  },
]

const MULTI_SCENARIOS: MemoryMultiScenario[] = [
  {
    ...makeBaseScenario('multi/contradiction', 'Session2 should update/delete contradicted session1 memory'),
    group: 'multi',
    sessions: [
      {
        transcript: loadTranscript('multi-contradiction-session1.txt'),
        currentMemory: richMem(
          ['Use TypeScript strict mode for all new files'],
          ['Always run the test suite before reporting a task as complete']
        ),
        expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['workflow'], allowedAdditionCategories: ['workflow'] },
      },
      {
        transcript: loadTranscript('multi-contradiction-session2.txt'),
        currentMemory: '',
        expected: { minTotalOps: 1, maxTotalOps: 2, expectUpdateOrDeletion: true },
      },
    ],
  },
  {
    ...makeBaseScenario('multi/already-captured', 'Session2 repeat should not duplicate existing memory'),
    group: 'multi',
    sessions: [
      {
        transcript: loadTranscript('multi-already-captured-session1.txt'),
        currentMemory: MEMORY_TEMPLATE,
        expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['codebase'], allowedAdditionCategories: ['codebase'] },
      },
      {
        transcript: loadTranscript('multi-already-captured-session2.txt'),
        currentMemory: '',
        expected: { expectEmpty: true, maxTotalOps: 0, forbidDuplicateOfExisting: true },
      },
    ],
  },
  {
    ...makeBaseScenario('multi/gradual-pattern', 'Second repetition should trigger codebase extraction'),
    group: 'multi',
    sessions: [
      {
        transcript: loadTranscript('multi-gradual-pattern-session1.txt'),
        currentMemory: MEMORY_TEMPLATE,
        expected: { expectEmpty: true, maxTotalOps: 0 },
      },
      {
        transcript: loadTranscript('multi-gradual-pattern-session2.txt'),
        currentMemory: '',
        expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['codebase'], allowedAdditionCategories: ['codebase'] },
      },
    ],
  },
  {
    ...makeBaseScenario('multi/refinement', 'Session2 should refine prior workflow preference'),
    group: 'multi',
    sessions: [
      {
        transcript: loadTranscript('multi-refinement-session1.txt'),
        currentMemory: richMem(
          ['Use snake_case for database column names, camelCase for TypeScript'],
          []
        ),
        expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['workflow'], allowedAdditionCategories: ['workflow'] },
      },
      {
        transcript: loadTranscript('multi-refinement-session2.txt'),
        currentMemory: '',
        expected: { minTotalOps: 1, maxTotalOps: 2, expectUpdateOrDeletion: true },
      },
    ],
  },
]

const QUALITY_SCENARIOS: MemorySingleScenario[] = [
  {
    ...makeBaseScenario('quality/concise-imperative-style', 'Extracted entries should be concise imperatives'),
    group: 'quality',
    transcript: loadTranscript('quality-concise-imperative-style.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { minTotalOps: 1, maxTotalOps: 2 },
    judgeChecks: [{ id: 'concise-imperative-style', description: 'Concise imperative statements', question: 'Are extracted memory entries concise imperative statements, not verbose narratives or explanations?' }],
  },
  {
    ...makeBaseScenario('quality/captures-primary-signal', 'Should capture primary durable signal in noise'),
    group: 'quality',
    transcript: loadTranscript('quality-captures-primary-signal.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { minTotalOps: 1, maxTotalOps: 3 },
    judgeChecks: [{ id: 'captures-primary-signal', description: 'Primary durable preference captured', question: 'Did extraction capture the most important durable user preference and omit irrelevant details?' }],
  },
  {
    ...makeBaseScenario('quality/no-overextraction', 'Should avoid overextracting task-local instructions'),
    group: 'quality',
    transcript: loadTranscript('quality-no-overextraction.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { minTotalOps: 1, maxTotalOps: 2 },
    judgeChecks: [{ id: 'no-overextraction', description: 'Avoid one-off detail extraction', question: 'Did it extract only the durable behavior-changing preference and avoid extracting one-off task instructions?' }],
  },
  {
    ...makeBaseScenario('quality/correct-update-not-duplicate', 'Outdated memory should be updated not duplicated'),
    group: 'quality',
    transcript: loadTranscript('quality-correct-update-not-duplicate.txt'),
    currentMemory: richMem(
      ['Use TypeScript strict mode for all new files', 'All API responses follow the { data, error, meta } envelope pattern'],
      ['Create a plan artifact before starting multi-file refactors']
    ),
    expected: { minTotalOps: 1, maxTotalOps: 2, expectUpdateOrDeletion: true },
    judgeChecks: [{ id: 'correct-update-not-duplicate', description: 'Update/replace outdated memory', question: 'Did it update or replace the outdated entry rather than adding a conflicting duplicate?' }],
  },
  {
    ...makeBaseScenario('quality/workflow-accuracy', 'Workflow extraction should capture subagent sequencing preference'),
    group: 'quality',
    transcript: loadTranscript('quality-workflow-accuracy.txt'),
    currentMemory: MEMORY_TEMPLATE,
    expected: { minTotalOps: 1, maxTotalOps: 2, requiredAdditionCategories: ['workflow'] },
    judgeChecks: [{ id: 'workflow-accuracy', description: 'Workflow preference accuracy', question: "Does the extracted workflow memory accurately capture the user's subagent/sequencing preference?" }],
  },
  {
    ...makeBaseScenario('quality/category-assignment', 'Category assignment should be correct across 2 categories'),
    group: 'quality',
    transcript: loadTranscript('quality-category-assignment.txt'),
    currentMemory: richMem(
      ['Run database migrations with prisma migrate before testing'],
      []
    ),
    expected: { minTotalOps: 2, maxTotalOps: 3, requiredAdditionCategories: ['codebase', 'workflow'] },
    judgeChecks: [{ id: 'category-assignment', description: 'Correct categories', question: 'Is each extracted memory item assigned to the correct category?' }],
  },
]

export const ALL_SCENARIOS: MemoryEvalScenario[] = [...DECISION_SCENARIOS, ...MULTI_SCENARIOS, ...QUALITY_SCENARIOS]

export const VARIANTS: EvalVariant[] = [
  { id: 'decision', label: 'Decision — objective write/no-write checks', count: DECISION_SCENARIOS.length },
  { id: 'multi', label: 'Multi — sequential multi-session extraction', count: MULTI_SCENARIOS.length },
  { id: 'quality', label: 'Quality — objective + judge quality checks', count: QUALITY_SCENARIOS.length },
]