import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { Effect } from 'effect'
import { defineTool } from '@magnitudedev/tools'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import { renderParseError } from '../presentation/error-render'
import type { RegisteredTool, ToolParseErrorEvent, TurnEngineEvent } from '../types'

const createTaskTool = defineTool({
  name: 'create_task',
  label: () => 'Create Task',
  group: 'tasks',
  description: 'Create a task',
  inputSchema: Schema.Struct({
    id: Schema.String,
    title: Schema.String,
    parent: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.Struct({ id: Schema.String }),
  execute: (input) => Effect.succeed({ id: input.id }),
})

function makeTools(): ReadonlyMap<string, RegisteredTool> {
  return new Map([
    ['create_task', { tool: createTaskTool as any, tagName: 'create_task', groupName: 'tasks' }],
  ])
}

function parse(input: string): TurnEngineEvent[] {
  const parser = createParser({ tools: makeTools() })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(['create_task']),
  )
  tokenizer.push(input + '\n')
  tokenizer.end()
  parser.end()
  return [...parser.drain()]
}

describe('parse error rendering for repeated same-tool invocations', () => {
  it('MissingRequiredField points at the correct (3rd) invocation, not the first', () => {
    const input = `<magnitude:invoke tool="create_task">
<magnitude:parameter name="id">phase1</magnitude:parameter>
<magnitude:parameter name="title">Phase 1: Project Setup</magnitude:parameter>
<magnitude:parameter name="parent">kanban</magnitude:parameter>
</magnitude:invoke>
<magnitude:invoke tool="create_task">
<magnitude:parameter name="id">phase2</magnitude:parameter>
<magnitude:parameter name="title">Phase 2: Data Layer & Stores</magnitude:parameter>
<magnitude:parameter name="parent">kanban</magnitude:parameter>
</magnitude:invoke>
<magnitude:invoke tool="create_task">
<magnitude:parameter name="id">phase3</magnitude:parameter>
<magnitude:parameter name">Phase 3: Board & Column UI</magnitude:parameter>
<magnitude:parameter name="parent">kanban</magnitude:parameter>
</magnitude:invoke>`

    const events = parse(input)
    const toolErrors = events.filter((e): e is ToolParseErrorEvent => e._tag === 'ToolParseError')
    const missingTitle = toolErrors.find(
      (e) => e.error._tag === 'MissingRequiredField' && e.error.parameterName === 'title',
    )

    expect(missingTitle).toBeDefined()

    const rendered = renderParseError(missingTitle!, input)

    expect(rendered).toContain("Missing required parameter 'title' for tool 'create_task'.")
    // Should point at the 3rd invocation (line 11), not the 1st (line 1)
    expect(rendered).toContain('11|<magnitude:invoke tool="create_task">')
    expect(rendered).toContain('12|<magnitude:parameter name="id">phase3</magnitude:parameter>')
    // Should NOT point at the 1st invocation
    expect(rendered).not.toContain('\n1|<magnitude:invoke tool="create_task">')
    expect(rendered).not.toContain('\n2|<magnitude:parameter name="id">phase1</magnitude:parameter>')
  })
})
