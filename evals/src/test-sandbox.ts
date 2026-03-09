/**
 * Test Sandbox
 * 
 * A test harness that wraps the real js-act sandbox (QuickJS WASM)
 * with mock tool implementations. Runs LLM responses through the
 * exact same pipeline as the real agent:
 * 
 * 1. Prose delimiter preprocessing (« » → JS string literals)
 * 2. Statement parsing (splitting code into executable statements)
 * 3. QuickJS sandbox execution (real JS engine, not regex)
 * 4. Tool dispatch with Schema validation
 * 
 * The only difference from production: tool execute() functions
 * capture their resolved arguments instead of performing real actions.
 */

import { Effect, Stream, Chunk } from 'effect'
import { Schema } from '@effect/schema'
import {
  Sandbox,
  createTool,
  createToolGroup,
  createJournal,
  ExecutionEvent,
  type SandboxItem,
  type SandboxOptions,
  type RuntimeOptions
} from '@magnitudedev/js-act'
import { PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE } from '@magnitudedev/agent'

// =============================================================================
// Types
// =============================================================================

/**
 * A captured tool call with its resolved (post-sandbox) arguments
 */
export interface CapturedCall {
  /** Tool group (e.g. 'default', 'fs', 'task') */
  group: string
  /** Tool name within group (e.g. 'message', 'write', 'shell') */
  name: string
  /** Fully qualified slug (e.g. 'default.message', 'fs.write') */
  slug: string
  /** The resolved input arguments as received by the tool's execute function */
  input: Record<string, unknown>
}

/**
 * Result of running code through the test sandbox
 */
export interface TestSandboxResult {
  /** All captured tool calls in execution order */
  calls: CapturedCall[]
  /** All raw ExecutionEvents from the sandbox */
  events: ExecutionEvent[]
  /** Execution error if any (sandbox crash, parse error, etc.) */
  error?: string
}

// =============================================================================
// Preprocessor config — identical to production
// =============================================================================

const PREPROCESSOR_CONFIG = {
  proseDelimiterOpen: PROSE_DELIM_OPEN,
  proseDelimiterClose: PROSE_DELIM_CLOSE,
  prosePatterns: [
    { pattern: 'message(', id: 'message' },
    { pattern: 'think(', id: 'think' }
  ]
} as const

// =============================================================================
// Mock tool factories
// =============================================================================

/**
 * Create the full set of mock tools that mirror the real agent's tool surface.
 * Every tool captures its resolved arguments into the shared `captured` array.
 */
function createMockTools(captured: CapturedCall[]): SandboxItem[] {
  
  function capture(group: string, name: string, input: Record<string, unknown>) {
    captured.push({ group, name, slug: `${group}.${name}`, input })
  }

  // =========================================================================
  // Default (global) tools
  // =========================================================================

  const mockMessage = createTool({
    name: 'message',
    description: 'Display a message to the user',
    inputSchema: Schema.Struct({ content: Schema.String }),
    outputSchema: Schema.String,
    argMapping: ['content'],
    execute: ({ content }) => {
      capture('default', 'message', { content })
      return Effect.succeed(content)
    },
  })

  const mockThink = createTool({
    name: 'think',
    description: 'Record internal reasoning',
    inputSchema: Schema.Struct({ thought: Schema.String }),
    outputSchema: Schema.String,
    argMapping: ['thought'],
    execute: ({ thought }) => {
      capture('default', 'think', { thought })
      return Effect.succeed(thought)
    },
  })

  const mockInspect = createTool({
    name: 'inspect',
    description: 'Examine a value',
    inputSchema: Schema.Struct({ target: Schema.Unknown }),
    outputSchema: Schema.String,
    argMapping: ['target'],
    execute: ({ target }) => {
      capture('default', 'inspect', { target })
      const json = typeof target === 'string' ? target : JSON.stringify(target, null, 2)
      return Effect.succeed(json)
    },
  })

  const mockDone = createTool({
    name: 'done',
    description: 'Yield turn to user',
    inputSchema: Schema.Struct({}),
    outputSchema: Schema.String,
    execute: () => {
      capture('default', 'done', {})
      return Effect.succeed('')
    },
  })

  const mockShell = createTool({
    name: 'shell',
    description: 'Run a shell command',
    inputSchema: Schema.Struct({ command: Schema.String }),
    outputSchema: Schema.Struct({
      stdout: Schema.String,
      stderr: Schema.String,
      exitCode: Schema.Number
    }),
    argMapping: ['command'],
    execute: ({ command }) => {
      capture('default', 'shell', { command })
      return Effect.succeed({ stdout: '', stderr: '', exitCode: 0 })
    }
  })

  const mockWebSearch = createTool({
    name: 'webSearch',
    description: 'Search the web',
    inputSchema: Schema.Struct({
      query: Schema.String,
      schema: Schema.optional(Schema.Unknown)
    }),
    outputSchema: Schema.Struct({
      text: Schema.String,
      sources: Schema.Array(Schema.Struct({ title: Schema.String, url: Schema.String }))
    }),
    argMapping: ['query', 'schema'],
    execute: ({ query, schema }) => {
      capture('default', 'webSearch', { query, schema })
      return Effect.succeed({ text: '', sources: [] })
    }
  })

  const mockSkill = createTool({
    name: 'skill',
    description: 'Activate a skill',
    inputSchema: Schema.Struct({ name: Schema.String }),
    outputSchema: Schema.Struct({
      name: Schema.String,
      content: Schema.String,
      source: Schema.String
    }),
    argMapping: ['name'],
    execute: ({ name }) => {
      capture('default', 'skill', { name })
      return Effect.succeed({ name, content: '', source: '' })
    }
  })

  const defaultGroup = createToolGroup({
    name: 'default',
    description: 'Built-in global tools',
    tools: [mockMessage, mockThink, mockInspect, mockDone, mockShell, mockWebSearch, mockSkill],
    global: true
  })

  // =========================================================================
  // Filesystem tools
  // =========================================================================

  const mockFsRead = createTool({
    name: 'read',
    description: 'Read file content',
    inputSchema: Schema.Struct({
      path: Schema.String,
      options: Schema.optional(Schema.Struct({
        lines: Schema.optional(Schema.Boolean)
      }))
    }),
    outputSchema: Schema.String,
    argMapping: ['path', 'options'],
    execute: ({ path, options }) => {
      capture('fs', 'read', { path, options })
      return Effect.succeed('mock file content')
    }
  })

  const mockFsWrite = createTool({
    name: 'write',
    description: 'Write content to file',
    inputSchema: Schema.Struct({
      path: Schema.String,
      content: Schema.String
    }),
    outputSchema: Schema.Void,
    argMapping: ['path', 'content'],
    execute: ({ path, content }) => {
      capture('fs', 'write', { path, content })
      return Effect.succeed(undefined as void)
    }
  })

  const EditSchema = Schema.Struct({
    from: Schema.String,
    to: Schema.optional(Schema.String),
    content: Schema.optional(Schema.String)
  })

  const mockFsEdit = createTool({
    name: 'edit',
    description: 'Edit file using hashline anchors',
    inputSchema: Schema.Struct({
      path: Schema.String,
      edits: Schema.Array(EditSchema)
    }),
    outputSchema: Schema.String,
    argMapping: ['path', 'edits'],
    execute: ({ path, edits }) => {
      capture('fs', 'edit', { path, edits })
      return Effect.succeed('Applied edits')
    }
  })

  const mockFsTree = createTool({
    name: 'tree',
    description: 'List directory structure',
    inputSchema: Schema.Struct({
      path: Schema.String,
      options: Schema.optional(Schema.Struct({
        recursive: Schema.optional(Schema.Boolean),
        maxDepth: Schema.optional(Schema.Number),
        gitignore: Schema.optional(Schema.Boolean)
      }))
    }),
    outputSchema: Schema.Array(Schema.Struct({
      path: Schema.String,
      name: Schema.String,
      type: Schema.String,
      depth: Schema.Number
    })),
    argMapping: ['path', 'options'],
    execute: ({ path, options }) => {
      capture('fs', 'tree', { path, options })
      return Effect.succeed([])
    }
  })

  const mockFsSearch = createTool({
    name: 'search',
    description: 'Search file contents',
    inputSchema: Schema.Struct({
      pattern: Schema.String,
      options: Schema.optional(Schema.Struct({
        path: Schema.optional(Schema.String),
        glob: Schema.optional(Schema.String)
      }))
    }),
    outputSchema: Schema.Array(Schema.Struct({
      file: Schema.String,
      match: Schema.String
    })),
    argMapping: ['pattern', 'options'],
    execute: ({ pattern, options }) => {
      capture('fs', 'search', { pattern, options })
      return Effect.succeed([])
    }
  })

  const fsGroup = createToolGroup({
    name: 'fs',
    description: 'Filesystem tools',
    tools: [mockFsRead, mockFsWrite, mockFsEdit, mockFsTree, mockFsSearch]
  })

  // =========================================================================
  // Task tools
  // =========================================================================

  const mockStartTask = createTool({
    name: 'startTask',
    description: 'Start a new work task',
    inputSchema: Schema.Struct({
      type: Schema.String,
      id: Schema.String,
      title: Schema.String
    }),
    outputSchema: Schema.String,
    argMapping: ['type', 'id', 'title'],
    execute: ({ type, id, title }) => {
      capture('task', 'startTask', { type, id, title })
      return Effect.succeed('Task started')
    }
  })

  const mockUpdateTask = createTool({
    name: 'updateTask',
    description: 'Update an existing task',
    inputSchema: Schema.Struct({
      id: Schema.String,
      updates: Schema.Unknown
    }),
    outputSchema: Schema.String,
    argMapping: ['id', 'updates'],
    execute: ({ id, updates }) => {
      capture('task', 'updateTask', { id, updates })
      return Effect.succeed('Task updated')
    }
  })

  const mockGetTask = createTool({
    name: 'getTask',
    description: 'Get task by ID',
    inputSchema: Schema.Struct({ id: Schema.String }),
    outputSchema: Schema.Unknown,
    argMapping: ['id'],
    execute: ({ id }) => {
      capture('task', 'getTask', { id })
      return Effect.succeed(null)
    }
  })

  const mockRequestBuild = createTool({
    name: 'requestBuild',
    description: 'Request transition to BUILD mode',
    inputSchema: Schema.Struct({ id: Schema.String }),
    outputSchema: Schema.String,
    argMapping: ['id'],
    execute: ({ id }) => {
      capture('task', 'requestBuild', { id })
      return Effect.succeed('Build requested')
    }
  })

  const mockValidate = createTool({
    name: 'validate',
    description: 'Run acceptance criteria',
    inputSchema: Schema.Struct({}),
    outputSchema: Schema.Unknown,
    execute: () => {
      capture('task', 'validate', {})
      return Effect.succeed({ results: [], allPassed: true })
    }
  })

  const taskGroup = createToolGroup({
    name: 'task',
    description: 'Task management tools',
    tools: [mockStartTask, mockUpdateTask, mockGetTask, mockRequestBuild, mockValidate],
    global: true
  })

  // =========================================================================
  // Fork tools (global scope)
  // =========================================================================

  const mockFork = createTool({
    name: 'fork',
    description: 'Create a background fork',
    inputSchema: Schema.Struct({
      name: Schema.String,
      params: Schema.Struct({
        prompt: Schema.String,
        taskId: Schema.optional(Schema.String)
      })
    }),
    outputSchema: Schema.Struct({ forkId: Schema.String }),
    argMapping: ['name', 'params'],
    execute: ({ name, params }) => {
      capture('default', 'fork', { name, params })
      return Effect.succeed({ forkId: 'mock-fork-id' })
    }
  })

  const mockSubmit = createTool({
    name: 'submit',
    description: 'Submit fork result',
    inputSchema: Schema.Struct({ result: Schema.String }),
    outputSchema: Schema.Void,
    argMapping: ['result'],
    execute: ({ result }) => {
      capture('default', 'submit', { result })
      return Effect.succeed(undefined as void)
    }
  })

  const mockForkSync = createTool({
    name: 'forkSync',
    description: 'Create a blocking fork',
    inputSchema: Schema.Struct({
      name: Schema.String,
      params: Schema.Struct({
        prompt: Schema.String,
        outputSchema: Schema.optional(Schema.Unknown)
      })
    }),
    outputSchema: Schema.Unknown,
    argMapping: ['name', 'params'],
    execute: ({ name, params }) => {
      capture('default', 'forkSync', { name, params })
      return Effect.succeed({ passed: true })
    }
  })

  return [defaultGroup, fsGroup, taskGroup, mockFork, mockSubmit, mockForkSync]
}

// =============================================================================
// Test Sandbox API
// =============================================================================

/**
 * Run a raw LLM response through the real js-act sandbox with mock tools.
 * 
 * Uses the exact same pipeline as the production agent:
 * - Prose delimiter preprocessing
 * - Statement parsing
 * - QuickJS WASM execution
 * - Schema-validated tool dispatch
 * 
 * Returns all captured tool calls with their fully resolved arguments.
 */
export async function runTestSandbox(rawResponse: string): Promise<TestSandboxResult> {
  const captured: CapturedCall[] = []
  const allEvents: ExecutionEvent[] = []
  const tools = createMockTools(captured)

  const program = Effect.scoped(
    Effect.gen(function* () {
      const journal = createJournal()
      const codeStream = Stream.make(rawResponse)

      const eventStream = Sandbox.stream(
        tools as unknown as readonly SandboxItem[],
        codeStream,
        {
          preprocessor: PREPROCESSOR_CONFIG,
          journal,
          observationTools: ['default.inspect'],
          nonToolTimeout: 10000
        }
      )

      yield* eventStream.pipe(
        Stream.runForEach((event: ExecutionEvent) => Effect.sync(() => {
          allEvents.push(event)
        }))
      )
    })
  )

  try {
    await Effect.runPromise(program as Effect.Effect<void, any>)
    return { calls: captured, events: allEvents }
  } catch (err) {
    return {
      calls: captured,
      events: allEvents,
      error: err instanceof Error ? err.message : String(err)
    }
  }
}

// =============================================================================
// Query helpers
// =============================================================================

/**
 * Find all captured calls matching a slug (e.g. 'fs.write', 'default.message')
 */
export function findCalls(result: TestSandboxResult, slug: string): CapturedCall[] {
  return result.calls.filter(c => c.slug === slug)
}

/**
 * Find all captured calls by group (e.g. 'fs', 'default', 'task')
 */
export function findCallsByGroup(result: TestSandboxResult, group: string): CapturedCall[] {
  return result.calls.filter(c => c.group === group)
}

/**
 * Get the first captured call matching a slug, or undefined
 */
export function findCall(result: TestSandboxResult, slug: string): CapturedCall | undefined {
  return result.calls.find(c => c.slug === slug)
}
