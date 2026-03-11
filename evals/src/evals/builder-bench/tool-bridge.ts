/**
 * Tool Bridge — SandboxTool definitions backed by Docker container operations.
 *
 * Creates real SandboxTool instances whose execute() functions delegate
 * to Docker CLI operations. Supports three toolset configurations.
 *
 * Builds a full AgentDefinition (with system prompt, turn policy, context policies)
 * that can be registered as an override for the real agent system.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createSandboxTool } from '@magnitudedev/js-act'
import { toolSet, defineAgent, continue_, finish, perceive, omit, defineThinkingLens } from '@magnitudedev/agent-definition'
import type { AgentDefinition, ToolSet } from '@magnitudedev/agent-definition'
import type { PolicyContext } from '@magnitudedev/agent'
import type { DockerContainer } from './docker'
import * as docker from './docker'

// =============================================================================
// Types
// =============================================================================

export type ToolsetId = 'fs-only' | 'shell-only' | 'fs-shell'

export interface ToolBridgeResult {
  /** Agent definition for the real agent system (register as override) */
  agentDef: AgentDefinition<ToolSet, PolicyContext>
}

// =============================================================================
// System Prompt (role description — protocol + tool docs added by Cortex)
// =============================================================================

const ROLE_DESCRIPTION = `You are a software engineer tasked with fixing bugs in a project.
You have access to tools for reading files, writing files, searching, and/or running shell commands.
Your goal is to diagnose and fix the bug so that all tests pass.

Rules:
- Do NOT modify any test files. Only fix the source code.
- Read the relevant source files and test files to understand the issue.
- After making changes, run the test command to verify your fix works.
- When all tests pass, call done() to signal completion.`

// =============================================================================
// Common Agent Config (shared between all toolset variants)
// =============================================================================

const BENCH_AGENT_CONFIG = {
  id: 'builder-bench-agent' as const,
  model: 'primary' as const,
  systemPrompt: ROLE_DESCRIPTION,
}

const benchThinkingLenses = [
  defineThinkingLens({
    name: 'quality',
    trigger: 'When writing or modifying code',
    description: "Consider code quality and adherence to existing patterns. Does this match the conventions, abstractions, and style already in use? Is this consistent with the surrounding codebase? Don't just make it work — make it fit.",
  }),
  defineThinkingLens({
    name: 'turn',
    trigger: 'When planning your next actions',
    description: 'Plan what to read and edit this turn. What files do you need to understand before making changes? What\'s the right order of edits?',
  }),
] as const

// =============================================================================
// Tool Factories
// =============================================================================

function createGlobalTools() {
  const messageTool = createSandboxTool({
    name: 'message',
    group: 'default',
    description: 'Display a message to the user',
    inputSchema: Schema.Struct({
      content: Schema.String.annotations({ description: 'Message content' }),
    }),
    outputSchema: Schema.String,
    argMapping: ['content'],
    bindings: { xmlInput: { type: 'tag', body: 'content' } } as const,
    execute: ({ content }) => Effect.succeed(content),
  })

  const thinkTool = createSandboxTool({
    name: 'think',
    group: 'default',
    description: 'Record internal reasoning (not shown to user)',
    inputSchema: Schema.Struct({
      thought: Schema.String.annotations({ description: 'Internal reasoning' }),
    }),
    outputSchema: Schema.String,
    argMapping: ['thought'],
    bindings: { xmlInput: { type: 'tag', body: 'thought' } } as const,
    execute: ({ thought }) => Effect.succeed(thought),
  })

  const doneTool = createSandboxTool({
    name: 'done',
    group: 'default',
    description: 'Signal that you have completed the task',
    inputSchema: Schema.Struct({}),
    outputSchema: Schema.String,
    bindings: { xmlInput: { type: 'tag', selfClosing: true } } as const,
    execute: () => Effect.succeed('Task completed.'),
  })

  return { messageTool, thinkTool, doneTool }
}

function createFsTools(container: DockerContainer) {
  const fsRead = createSandboxTool({
    name: 'read',
    group: 'fs',
    description: 'Read file content as string',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({ description: 'Relative path from workspace root' }),
    }),
    outputSchema: Schema.String,
    argMapping: ['path'],
    bindings: { xmlInput: { type: 'tag', attributes: ['path'], selfClosing: true } } as const,
    execute: ({ path }) => Effect.promise(() => docker.readFile(container, path)),
  })

  const fsWrite = createSandboxTool({
    name: 'write',
    group: 'fs',
    description: 'Write content to file (creates parent directories if needed)',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({ description: 'Relative path from workspace root' }),
      content: Schema.String.annotations({ description: 'File content to write' }),
    }),
    outputSchema: Schema.Void,
    argMapping: ['path', 'content'],
    bindings: { xmlInput: { type: 'tag', attributes: ['path'], body: 'content' } } as const,
    execute: ({ path, content }) => Effect.promise(async () => {
      const dir = path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : null
      if (dir) {
        await docker.execInContainer(container, `mkdir -p ${JSON.stringify(dir)}`)
      }
      await docker.writeFile(container, path, content)
    }),
  })

  const fsTree = createSandboxTool({
    name: 'tree',
    group: 'fs',
    description: 'List directory structure',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({ description: 'Relative path from workspace root' }),
      options: Schema.optional(Schema.Struct({
        recursive: Schema.optional(Schema.Boolean),
        maxDepth: Schema.optional(Schema.Number),
      })),
    }),
    outputSchema: Schema.Array(Schema.Struct({
      path: Schema.String,
      name: Schema.String,
      type: Schema.String,
      depth: Schema.Number,
    })),
    argMapping: ['path', 'options'],
    execute: ({ path, options }) => Effect.promise(() =>
      docker.listDir(container, path, options)
    ),
  })

  const fsSearch = createSandboxTool({
    name: 'search',
    group: 'fs',
    description: 'Search file contents with regex',
    inputSchema: Schema.Struct({
      pattern: Schema.String.annotations({ description: 'Regex pattern to search for' }),
      options: Schema.optional(Schema.Struct({
        path: Schema.optional(Schema.String),
        glob: Schema.optional(Schema.String),
      })),
    }),
    outputSchema: Schema.Array(Schema.Struct({
      file: Schema.String,
      match: Schema.String,
    })),
    argMapping: ['pattern', 'options'],
    execute: ({ pattern, options }) => Effect.promise(() =>
      docker.searchFiles(container, pattern, options?.path ?? '.', options?.glob)
    ),
  })

  return { fsRead, fsWrite, fsTree, fsSearch }
}

function createShellTool(container: DockerContainer) {
  return createSandboxTool({
    name: 'shell',
    group: 'default',
    description: 'Execute a shell command',
    inputSchema: Schema.Struct({
      command: Schema.String.annotations({ description: 'Shell command to execute' }),
    }),
    outputSchema: Schema.Struct({
      stdout: Schema.String,
      stderr: Schema.String,
      exitCode: Schema.Number,
    }),
    argMapping: ['command'],
    bindings: { xmlInput: { type: 'tag', body: 'command' } } as const,
    execute: ({ command }) => Effect.promise(() =>
      docker.execInContainer(container, command, 60_000)
    ),
  })
}

// =============================================================================
// Context Policies
// =============================================================================

const omitCtx = () => omit()

const shellCtx = (input: { command: string }, output: { stdout: string; stderr: string; exitCode: number }) => {
  const parts = [`$ ${input.command}`]
  if (output.stdout) parts.push(output.stdout.trimEnd())
  if (output.stderr) parts.push(`STDERR: ${output.stderr.trimEnd()}`)
  parts.push(`[exit: ${output.exitCode}]`)
  return perceive(parts.join('\n'))
}

const passthroughCtx = (_input: unknown, output: string) => perceive(output)
const structuredCtx = (_input: unknown, output: unknown) =>
  perceive(typeof output === 'string' ? output : JSON.stringify(output))
const writeCtx = () => perceive('File written.')

// =============================================================================
// Tool Bridge Factory — one defineAgent per toolset for type safety
// =============================================================================

function createFsOnlyAgent(container: DockerContainer): AgentDefinition<ToolSet, PolicyContext> {
  const { messageTool, thinkTool, doneTool } = createGlobalTools()
  const { fsRead, fsWrite, fsTree, fsSearch } = createFsTools(container)

  const tools = toolSet({ message: messageTool, think: thinkTool, done: doneTool, fileRead: fsRead, fileWrite: fsWrite, fileTree: fsTree, fileSearch: fsSearch })
  return defineAgent(tools, {
    ...BENCH_AGENT_CONFIG,
    thinkingLenses: benchThinkingLenses.slice(),
    permission: (p) => ({ _default: () => p.allow() }),
    turn: { decide: (turnCtx) => turnCtx.cancelled || turnCtx.toolsCalled.includes('done') ? finish() : continue_() },
    context: { message: omitCtx, think: omitCtx, done: omitCtx, fileRead: passthroughCtx, fileWrite: writeCtx, fileTree: structuredCtx, fileSearch: structuredCtx },
    display: (d) => ({ think: () => d.hidden(), _default: () => d.visible() }),
  })
}

function createShellOnlyAgent(container: DockerContainer): AgentDefinition<ToolSet, PolicyContext> {
  const { messageTool, thinkTool, doneTool } = createGlobalTools()
  const shell = createShellTool(container)

  const tools = toolSet({ message: messageTool, think: thinkTool, done: doneTool, shell })
  return defineAgent(tools, {
    ...BENCH_AGENT_CONFIG,
    thinkingLenses: benchThinkingLenses.slice(),
    permission: (p) => ({ _default: () => p.allow() }),
    turn: { decide: (turnCtx) => turnCtx.cancelled || turnCtx.toolsCalled.includes('done') ? finish() : continue_() },
    context: { message: omitCtx, think: omitCtx, done: omitCtx, shell: shellCtx },
    display: (d) => ({ think: () => d.hidden(), _default: () => d.visible() }),
  })
}

function createFsShellAgent(container: DockerContainer): AgentDefinition<ToolSet, PolicyContext> {
  const { messageTool, thinkTool, doneTool } = createGlobalTools()
  const { fsRead, fsWrite, fsTree, fsSearch } = createFsTools(container)
  const shell = createShellTool(container)

  const tools = toolSet({ message: messageTool, think: thinkTool, done: doneTool, shell, fileRead: fsRead, fileWrite: fsWrite, fileTree: fsTree, fileSearch: fsSearch })
  return defineAgent(tools, {
    ...BENCH_AGENT_CONFIG,
    thinkingLenses: benchThinkingLenses.slice(),
    permission: (p) => ({ _default: () => p.allow() }),
    turn: { decide: (turnCtx) => turnCtx.cancelled || turnCtx.toolsCalled.includes('done') ? finish() : continue_() },
    context: { message: omitCtx, think: omitCtx, done: omitCtx, shell: shellCtx, fileRead: passthroughCtx, fileWrite: writeCtx, fileTree: structuredCtx, fileSearch: structuredCtx },
    display: (d) => ({ think: () => d.hidden(), _default: () => d.visible() }),
  })
}

/**
 * Create Docker-bridged tools for a given toolset configuration.
 * Returns an agent definition suitable for registering as an override.
 */
export function createDockerTools(
  container: DockerContainer,
  toolsetId: ToolsetId,
): ToolBridgeResult {
  switch (toolsetId) {
    case 'fs-only':
      return { agentDef: createFsOnlyAgent(container) }
    case 'shell-only':
      return { agentDef: createShellOnlyAgent(container) }
    case 'fs-shell':
      return { agentDef: createFsShellAgent(container) }
  }
}
