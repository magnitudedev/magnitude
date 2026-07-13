/**
 * Example: Defining a typed harness tool
 *
 * Shows defineHarnessTool with typed input/output/error schemas,
 * and demonstrates the never-switch generic pattern on event types.
 */

import { Schema, Effect } from 'effect'
import { defineHarnessTool } from '../src'
import type {
  ToolResult,
  ToolExecutionStarted,
  ToolExecutionEnded,
  ToolLifecycleEvent,
  HarnessEvent,
  ToolError,
} from '../src'

// ── Schemas ──────────────────────────────────────────────────────────

const FileReadInput = Schema.Struct({
  path: Schema.String.annotations({ description: 'Relative path to file' }),
  offset: Schema.optional(Schema.Number).annotations({ description: '1-indexed start line' }),
  limit: Schema.optional(Schema.Number).annotations({ description: 'Max lines to return' }),
})

const FileReadOutput = Schema.String

const FileReadError = Schema.Struct({
  _tag: Schema.Literal('FsError'),
  message: Schema.String,
})

// Derive TypeScript types from schemas
type FileReadInput = typeof FileReadInput.Type
type FileReadOutput = typeof FileReadOutput.Type
type FileReadError = typeof FileReadError.Type

// ── Define the tool ──────────────────────────────────────────────────

const readTool = defineHarnessTool({
  definition: {
    name: 'read',
    description: 'Read a file',
    inputSchema: FileReadInput,
    outputSchema: FileReadOutput,
  },
  errorSchema: FileReadError,
  execute: ({ path, offset, limit }) =>
    Effect.succeed(`contents of ${path} from line ${offset ?? 1}`),
})

// readTool is HarnessToolConcrete<FileReadInput, string, never, FsError, never>
// readTool.definition.inputSchema is Schema.Schema<FileReadInput, ...>

// ── Never-switch generics on event types ─────────────────────────────

// Bare type (no generic args) → erased: all payloads are `any`
type ErasedResult = ToolResult
//   = { _tag: "Success"; output: any } | { _tag: "Error"; error: any } | ...

// Parameterized type → concrete: payloads are typed
type ConcreteResult = ToolResult<string, FileReadError>
//   = { _tag: "Success"; output: string } | { _tag: "Error"; error: FileReadError } | ...

// Same pattern on lifecycle events
type ErasedEvent = ToolLifecycleEvent
//   input/output/emission/error all erased to `any`

type ConcreteEvent = ToolLifecycleEvent<FileReadInput, string, never, FileReadError>
//   ToolExecutionStarted has input: FileReadInput
//   ToolExecutionEnded has result: ToolResult<string, FileReadError>

// And on HarnessEvent
type ErasedHarnessEvent = HarnessEvent
type ConcreteHarnessEvent = HarnessEvent<FileReadInput, string, never, FileReadError>

// ── Usage in a handler ───────────────────────────────────────────────

function handleErasedEvent(event: ToolLifecycleEvent) {
  // Erased — you can switch on _tag but payloads are `any`
  if (event._tag === 'ToolExecutionEnded') {
    const result = event.result // ToolResult (erased)
    if (result._tag === 'Success') {
      console.log(result.output) // any
    }
  }
}

function handleConcreteEvent(event: ToolLifecycleEvent<FileReadInput, string, never, FileReadError>) {
  if (event._tag === 'ToolExecutionEnded') {
    const result = event.result // ToolResult<string, FileReadError>
    if (result._tag === 'Success') {
      console.log(result.output) // string
    }
    if (result._tag === 'Error') {
      console.log(result.error.message) // string (from FileReadError)
    }
  }
}
