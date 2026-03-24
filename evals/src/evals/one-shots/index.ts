/**
 * One-Shots — standalone scenario evals that call primary.chat directly.
 *
 * Each scenario is a self-contained { systemPrompt, messages } pair.
 * The eval calls callModel(systemPrompt, messages) and runs checks on the raw response.
 */

import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, Check, CheckResult } from '../../types'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { callModel } from '../../runner'

// =============================================================================
// Scenario definitions
// =============================================================================

interface OneShotDef {
  id: string
  description: string
  systemPrompt: string
  messages: ChatMessage[]
  checks: Check[]
}

const SCENARIOS: OneShotDef[] = [
  {
    id: 'explorer-long-artifact',
    description: 'Explorer agent writes a 200+ line artifact about collaborative editors',
    systemPrompt: `# XML-ACT Protocol

You respond using the XML-ACT protocol to think, message the user, and take actions with tools.
XML-ACT isn't a strict XML format - no HTML entity escaping or other XML escaping is required - it only parses the specific tags described here.

## Required Turn format
<think>...concise or deep thinking</think>
...message for user, required if they just messaged you
<actions>
...zero or more actions
<inspect>
...one or more refs
</inspect>
</actions>

- Think block is required, even if brief
- Message is optional, unless the user just messaged you
- Action block is optional if you are just messaging the user

## Core Rules
- Use only defined tags — all other text is treated as literal content, not markup
- Quote attribute values
- Prefer attributes for scalar params
- Put long/free-form strings in element body text
- Do NOT escape HTML entities or special characters (\`&\`, \`<\`, \`>\`, etc.) anywhere — in prose, in element body text, in attribute values. Only the defined tags are parsed; all other text, including angle-bracket sequences that look like tags, passes through verbatim.
- If user just sent a message, you must include a message to them in your next response to keep them in the loop, even if also taking actions

## Action Calls
Inside \`<actions>\`, each tool call is an XML element. Give each call an \`id\` attribute to enable composition.
\`<search id="s1" query="weather" />\`
\`<fs-write id="w1" path="/tmp/a.txt">long content...</fs-write>\`

## Composition (refs)
Tool outputs are XML-serialized. Use \`<ref id=".."/>\` inside a tool body to insert a prior action's output inline.

Without \`query\`, the ref resolves to the full text content of the output:
\`\`\`
<fs-read id="r1" path="src/config.ts" />
<edit id="e1" path="src/config.ts">
<old>const port = 3000;</old>
<new>const port = 8080;</new>
</edit>
\`\`\`

With \`query\`, an XPath/XQuery 3.1 expression selects into the XML output:
\`\`\`
<search id="s1" pattern="TODO" />
<fs-read id="r1" path="...">
<ref id="s1" query="//item[1]/@file" />
</fs-read>
\`\`\`

Query examples:
- \`//item[1]\` — first item element
- \`//item/@file\` — file attribute from all items
- \`count(//item)\` — count items
- \`content\` — select a specific child element by name

If a query returns no results or errors, the full text content is returned as fallback.

## Progressive Disclosure
At the end of \`<actions>\`, add an optional \`<inspect>\` block to choose which results you see.
Use \`<ref>\` entries to select specific outputs. Without \`query\`, you see the full output; with \`query\`, you see a subset:
\`\`\`
<inspect>
<ref id="s1" />
<ref id="r1" query="//item[1]" />
</inspect>
\`\`\`

## Special Tags
\`<think>...</think>\` internal reasoning (not shown to user)

## Example Turn
<think>Need to find the config file and update the port.</think>
Updating the port configuration.
<actions>
<search id="s1" pattern="port" path="src/" />
<fs-read id="r1">
<ref id="s1" query="//item[1]/@file" />
</fs-read>
<edit id="e1" path="src/config.ts">
<old>const port = 3000;</old>
<new>const port = 8080;</new>
</edit>
<inspect>
<ref id="s1" />
<ref id="r1" query="content" />
</inspect>
</actions>


# Explorer

You are an explorer agent. Your job is to go deep — thoroughly understand a specific topic, mechanic, or problem space and produce a synthesized analysis.

## Use Cases

You are deployed when the team lead needs:
- **Deep codebase analysis** — understanding how a specific system or mechanism works in detail, tracing through layers, building a complete mental model
- **External research** — finding information about libraries, APIs, techniques, or best practices via web search
- **Ideation** — exploring an open-ended problem space, brainstorming approaches, and converging on a well-reasoned solution

## Methodology

1. **Understand what you're being asked** — Read your task carefully. What specifically does the team lead need to know or figure out?
2. **Go deep, not broad** — Your value is thoroughness. Trace through systems completely. Follow chains of calls. Read the actual implementations, not just the interfaces.
3. **Synthesize, don't just report** — Don't dump raw findings. Build a mental model, identify the key insights, and present a coherent understanding.
4. **Think hard** — Use \`<think>\` liberally. For ideation tasks especially, work through multiple angles internally before committing to a direction.

## Output

- Write your findings to an artifact.
- When done, use \`<parent-message>\` with a summary of your key findings and conclusions.
- If you discover something unexpected or important beyond your assignment, message the team lead.

## Principles

- Depth over breadth. You don't just map terrain; you drill into one area and understand it completely.
- Ground everything in evidence. Reference specific files, functions, types, code paths. Quote relevant code when it clarifies a point.
- For ideation: don't just list options. Actively reason through trade-offs, eliminate weak approaches, and converge on a recommendation with clear rationale.


## Tools

### fs

Read file content as string

<fs-read id="r1"
path="..." <!-- — Relative path to a file from cwd. Use tree instead for directories -->
/>

Returns: string
  <fs-read>...</fs-read>

List directory structure with optional gitignore filtering

<fs-tree id="r1"
path="..." <!-- — Relative path from cwd -->
/>

Returns:
  <fs-tree />

Search file contents with regex

<fs-search id="r1"
pattern="..." <!-- — Regex pattern to search for -->
path="..." <!-- optional. — Directory to search in (default: cwd) -->
glob="..." <!-- optional. — Glob pattern to filter files (e.g., "*.ts") -->
limit="..." <!-- optional. number. — Maximum number of matches to return (default: 50) -->
/>

Returns:
  <fs-search />

### Global

Execute a shell command

<shell id="r1">command</shell>
<!-- command (required, body) — a string -->

Returns:
  <shell>
    <stdout>stdout</stdout>
    <stderr>stderr</stderr>
    <exitCode>exitCode</exitCode> <!-- number -->
  </shell>

Search the web and optionally extract structured data

<webSearch id="r1">query</webSearch>
<!-- query (required, body) — a string -->

Returns:
  <webSearch />

### artifact

Read the current content of an artifact.

<artifact-read id="r1"
name="..." <!-- — Artifact name -->
/>

Returns: string
  <artifact-read>...</artifact-read>

Write content to an artifact (full replace).

<artifact-write id="r1"
name="..." <!-- — Artifact name -->
>content</artifact-write>
<!-- content (required, body) — New content for the artifact -->

Returns: string
  <artifact-write>...</artifact-write>

### parent

Send a message to the team lead. Use this to report progress, ask questions, deliver results, or flag concerns. You will go idle until the team lead responds.

<parent-message id="r1">content</parent-message>
<!-- content (required, body) — Message content (markdown) -->

Returns: string
  <parent-message>...</parent-message>`,
    messages: [
      {
        role: 'user',
        content: [
          `<project_context>
Working directory: /Users/trg/magnitude-secret
Shell: zsh
Username: trg
Platform: macos
Git branch: main
Git status:
M cli/src/app.tsx
 M cli/src/components/artifact-reader-panel.tsx
 M cli/src/components/artifact-ref-chip.tsx
 M cli/src/components/assistant-message.tsx
 M cli/src/components/inline-fork-activity.tsx
 M cli/src/components/markdown-content.tsx
 M cli/src/components/message-view.tsx
 M cli/src/components/proposal-banner.tsx
 M cli/src/components/think-block.tsx
 M cli/src/utils/artifact-refs.tsx
 M cli/src/visuals/define.ts
 M packages/agent/src/agents/prompts/lead.txt

Folder structure:
evals/ (~515k tok)
cli/ (~215k tok)
packages/ (~990k tok)
specs/ (~537k tok)
bugs/ (~58k tok)
docs/ (~59k tok)

<agentfile filename="AGENTS.md">
# Magnitude Agent Instructions

You are working on Magnitude, an AI coding agent platform.
</agentfile>
</project_context>

<lead>
<title>Write a long detailed artifact about building a real-time collaborative editor</title>
<prompt>
Write a detailed artifact called "collab-editor-guide" — a comprehensive guide to building a real-time collaborative editor.

Requirements:
- At least 200 lines of markdown
- At least 8-10 distinct sections with ## headings, and several ### sub-headings
- Include code examples (TypeScript/JavaScript), bullet lists, and substantial prose under each heading
- Cover topics like: Architecture Overview, CRDT Fundamentals, Operational Transform vs CRDTs, WebSocket Layer, Conflict Resolution, Presence and Cursors, Undo/Redo, Rich Text Support, Testing Strategies, Deployment and Scaling

Make it realistic and detailed. This is for UI testing so quality matters for realism but accuracy is not critical.

Write the artifact and then message me when done.
</prompt>
</lead>`,
        ],
      },
      {
        role: 'assistant',
        content: [
          `<think>I am a background agent spawned to complete a specific task. The conversation history above is inherited context from my parent agent, not my own. I will focus on the task described above and submit results when done.</think>`,
        ],
      },
      {
        role: 'user',
        content: [
          `<results>
<reminder>Focus on completing your assigned task. Use submit() when done.</reminder>
</results>`,
        ],
      },
    ],
    checks: [
      {
        id: 'think',
        description: 'Response has opening and closing <think> tags',
        evaluate(raw) {
          const hasOpen = raw.includes('<think>')
          const hasClose = raw.includes('</think>')
          const passed = hasOpen && hasClose
          return { passed, message: passed ? undefined : `Missing ${!hasOpen ? '<think>' : '</think>'}` }
        },
      },
      {
        id: 'actions',
        description: 'Response has opening and closing <actions> tags',
        evaluate(raw) {
          const hasOpen = raw.includes('<actions>')
          const hasClose = raw.includes('</actions>')
          const passed = hasOpen && hasClose
          return { passed, message: passed ? undefined : `Missing ${!hasOpen ? '<actions>' : '</actions>'}` }
        },
      },
      {
        id: 'inspect',
        description: 'Response has opening and closing <inspect> tags',
        evaluate(raw) {
          const hasOpen = raw.includes('<inspect>')
          const hasClose = raw.includes('</inspect>')
          const passed = hasOpen && hasClose
          return { passed, message: passed ? undefined : `Missing ${!hasOpen ? '<inspect>' : '</inspect>'}` }
        },
      },
      {
        id: 'artifact-write',
        description: 'Response has opening and closing <artifact-write> tags',
        evaluate(raw) {
          const hasOpen = /<artifact-write[\s>]/.test(raw)
          const hasClose = raw.includes('</artifact-write>')
          const passed = hasOpen && hasClose
          return { passed, message: passed ? undefined : `Missing ${!hasOpen ? '<artifact-write>' : '</artifact-write>'}` }
        },
      },
    ],
  },
]

// =============================================================================
// Build eval scenarios
// =============================================================================

function buildScenarios(): Scenario[] {
  return SCENARIOS.map((def) => ({
    id: def.id,
    description: def.description,
    messages: def.messages,
    checks: def.checks,
  }))
}

// =============================================================================
// Execution
// =============================================================================

function makeFail(scenarioId: string, message: string, rawResponse: string = ''): ScenarioResult {
  const checks: Record<string, CheckResult> = {}
  const scenarioDef = SCENARIOS.find((s) => s.id === scenarioId)
  if (scenarioDef) {
    for (const check of scenarioDef.checks) {
      checks[check.id] = { passed: false, message }
    }
  }
  return { scenarioId, checks, passed: false, score: 0, rawResponse }
}

async function executeScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  const def = SCENARIOS.find((s) => s.id === scenario.id)
  if (!def) return makeFail(scenario.id, `Unknown scenario: ${scenario.id}`)

  let rawResponse: string
  try {
    rawResponse = await callModel(def.systemPrompt, def.messages, modelSpec)
  } catch (error) {
    return makeFail(scenario.id, `Model call failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  const checks: Record<string, CheckResult> = {}
  let allPassed = true
  for (const check of def.checks) {
    const result = check.evaluate(rawResponse, null as never)
    checks[check.id] = result
    if (!result.passed) allPassed = false
  }

  const scores = Object.values(checks)
  const avgScore = scores.length > 0
    ? scores.reduce((sum, c) => sum + (c.score ?? (c.passed ? 1 : 0)), 0) / scores.length
    : 0

  return {
    scenarioId: scenario.id,
    checks,
    passed: allPassed,
    score: avgScore,
    rawResponse,
  }
}

// =============================================================================
// Export
// =============================================================================

const scenarios = buildScenarios()

export const oneShotsEval: RunnableEval = {
  id: 'one-shots',
  name: 'One-Shot Scenarios',
  description: `Standalone scenarios that call primary.chat directly (${scenarios.length} scenarios)`,
  scenarios,
  defaultConcurrency: 4,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario, modelSpec)
  },
}
