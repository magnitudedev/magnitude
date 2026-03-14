/**
 * Telemetry Event Helpers — typed capture functions for each telemetry event.
 *
 * Each function maps to a PostHog event with defined properties.
 * All properties are anonymous aggregate data — no personal content.
 */

import { capture } from './client'

// ========== Session Lifecycle ==========

export function trackSessionStart(props: {
  platform: string
  shell: string
  isResume: boolean
}): void {
  capture('session_start', props)
}

export function trackSessionEnd(props: {
  durationSeconds: number
  totalTurns: number
  totalTools: number
  totalUserMessages: number
  totalInputTokens: number
  totalOutputTokens: number
  totalLinesWritten: number
  totalLinesAdded: number
  totalLinesRemoved: number
  agentCount: number
}): void {
  capture('session_end', props)
}

// ========== User Messages ==========

export function trackUserMessage(props: {
  mode: 'text' | 'audio'
  synthetic: boolean
  taskMode: boolean
  hasAttachments: boolean
}): void {
  capture('user_message', props)
}

// ========== Turn Completed (LLM Call) ==========

export function trackTurnCompleted(props: {
  providerId: string | null
  modelId: string | null
  modelSlot: 'primary' | 'secondary' | 'browser'
  authType: string | null
  inputTokens: number | null
  outputTokens: number | null
  cacheReadTokens: number | null
  cacheWriteTokens: number | null
  toolCount: number
  success: boolean
  forkId: string | null
  agentRole: string
}): void {
  capture('turn_completed', props)
}

// ========== Tool Usage ==========

export function trackToolUsage(props: {
  toolName: string
  group: string
  status: 'success' | 'error' | 'rejected' | 'interrupted'
  linesAdded?: number
  linesRemoved?: number
  linesWritten?: number
  forkId: string | null
  agentRole: string
}): void {
  capture('tool_usage', props)
}

// ========== Agent Lifecycle ==========

export function trackAgentSpawned(props: {
  agentType: string
  mode: 'clone' | 'spawn'
}): void {
  capture('agent_spawned', props)
}

export function trackAgentCompleted(props: {
  agentType: string
  durationSeconds: number
}): void {
  capture('agent_completed', props)
}

// ========== Provider ==========

export function trackProviderConnected(props: {
  providerId: string
  authType: string
}): void {
  capture('provider_connected', props)
}

// ========== Compaction ==========

export function trackCompaction(props: {
  tokensSaved: number
  success: boolean
}): void {
  capture('compaction', props)
}
