import { Effect } from 'effect'
import type { ChatMessage, ExtractMemoryDiffResult } from '@magnitudedev/llm-core'
import type { CallUsage } from '../state/provider-state'
import { type StreamingFn, type CompleteFn, type BoundModel, type ChatStream } from './bound-model'

function includesClaudeSpoof(model: BoundModel): boolean {
  return model.model.providerId === 'anthropic' && model.connection.auth?.type === 'oauth'
}

function isCodexPath(model: BoundModel): boolean {
  const auth = model.connection.auth
  return (
    (model.model.providerId === 'openai' && auth?.type === 'oauth') ||
    (model.model.providerId === 'github-copilot' && model.model.id.includes('codex'))
  )
}

function codexCallOptions(model: BoundModel, systemPrompt: string) {
  if (!isCodexPath(model)) return undefined
  return {
    providerOptions: {
      instructions: systemPrompt,
      ...(model.model.providerId === 'openai' ? { store: false } : {}),
    },
  }
}

function includeSystemPromptMessage(model: BoundModel): boolean {
  return !isCodexPath(model)
}

export const CodingAgentChat: StreamingFn<
  { systemPrompt: string; messages: ChatMessage[]; options?: { stopSequences?: string[] }; ackTurn: string },
  ChatStream
> = {
  name: 'CodingAgentChat',
  mode: 'stream',
  execute: (model, input) =>
    model.stream(
      'CodingAgentChat',
      [input.systemPrompt, input.messages, input.ackTurn, includesClaudeSpoof(model), includeSystemPromptMessage(model)],
      {
        stopSequences: input.options?.stopSequences,
        ...(codexCallOptions(model, input.systemPrompt) ?? {}),
      },
    ),
}

export const SimpleChat: StreamingFn<
  { systemPrompt: string; messages: ChatMessage[]; options?: { stopSequences?: string[] } },
  ChatStream
> = {
  name: 'SimpleChat',
  mode: 'stream',
  execute: (model, input) =>
    model.stream(
      'SimpleChat',
      [input.systemPrompt, input.messages, includeSystemPromptMessage(model)],
      {
        stopSequences: input.options?.stopSequences,
        ...(codexCallOptions(model, input.systemPrompt) ?? {}),
      },
    ),
}

export const CodingAgentCompact: CompleteFn<
  { systemPrompt: string; messages: ChatMessage[] },
  { text: string; usage: CallUsage }
> = {
  name: 'CodingAgentCompact',
  mode: 'complete',
  execute: (model, input) =>
    Effect.map(
      model.complete(
        'CodingAgentCompact',
        [input.systemPrompt, input.messages, includesClaudeSpoof(model), includeSystemPromptMessage(model)],
        codexCallOptions(model, input.systemPrompt),
      ),
      ({ result, usage }) => ({ text: result, usage }),
    ),
}

export const ExtractMemoryDiff: CompleteFn<
  { transcript: string; currentMemory: string },
  ExtractMemoryDiffResult
> = {
  name: 'ExtractMemoryDiff',
  mode: 'complete',
  execute: (model, input) =>
    Effect.map(
      model.complete('ExtractMemoryDiff', [input.transcript, input.currentMemory, includesClaudeSpoof(model)]),
      ({ result }) => result,
    ),
}

export const GatherSplit: CompleteFn<
  { query: string; fileTree: string; tokenBudget: number },
  { result: { path: string; query: string }[]; usage: CallUsage }
> = {
  name: 'GatherSplit',
  mode: 'complete',
  execute: (model, input) =>
    Effect.map(
      model.complete('GatherSplit', [input.query, input.fileTree, input.tokenBudget, includesClaudeSpoof(model)]),
      ({ result, usage }) => ({ result, usage }),
    ),
}

export const PatchFile: CompleteFn<
  { instructions: string; fileContent: string; previousAttempts?: Array<{ response: string; error: string }> },
  { result: string; usage: CallUsage }
> = {
  name: 'PatchFile',
  mode: 'complete',
  execute: (model, input) =>
    Effect.map(
      model.complete('PatchFile', [
        input.instructions,
        input.fileContent,
        input.previousAttempts ?? [],
        includesClaudeSpoof(model),
      ]),
      ({ result, usage }) => ({ result, usage }),
    ),
}

export const CreateFile: CompleteFn<
  { instructions: string; filePath: string },
  { result: string; usage: CallUsage }
> = {
  name: 'CreateFile',
  mode: 'complete',
  execute: (model, input) =>
    Effect.map(
      model.complete('CreateFile', [input.instructions, input.filePath, includesClaudeSpoof(model)]),
      ({ result, usage }) => ({ result, usage }),
    ),
}

export const AutopilotContinuation: CompleteFn<
  { systemPrompt: string; messages: ChatMessage[] },
  { result: string; usage: CallUsage }
> = {
  name: 'AutopilotContinuation',
  mode: 'complete',
  execute: (model, input) =>
    Effect.map(
      model.complete(
        'AutopilotContinuation',
        [input.systemPrompt, input.messages, includesClaudeSpoof(model), includeSystemPromptMessage(model)],
        codexCallOptions(model, input.systemPrompt),
      ),
      ({ result, usage }) => ({ result, usage }),
    ),
}