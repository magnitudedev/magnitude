/**
 * ATIF trajectory serialization
 *
 * Converts the accumulated AtifProjectionState into a valid ATIF v1.7 JSON object.
 */

import { Option } from 'effect'
import type {
  AtifTrajectory,
  AtifForkState,
  AtifFinalMetrics,
  AtifAgent,
} from './types'
import { toolDefinitionsFromToolkit } from './tool-definitions'
import type { ConfigState } from '../../ambient/config-ambient'
import type { JsonValue } from '@magnitudedev/ai'

// =============================================================================
// Helpers
// =============================================================================

function forkToTrajectory(
  fork: AtifForkState,
  trajectoryId: string,
  sessionId: string | undefined,
  configState: ConfigState,
  solo?: boolean,
): AtifTrajectory {
  const metrics = fork.tokenAccumulator
  const finalMetrics: AtifFinalMetrics | undefined = fork.steps.length > 0
      ? {
          total_prompt_tokens: Option.some(metrics.promptTokens),
          total_completion_tokens: Option.some(metrics.completionTokens),
          total_cached_tokens: Option.some(metrics.cachedTokens),
          total_cost_usd: Option.some(metrics.costUsd),
          total_steps: Option.some(fork.steps.length),
          extra: Option.none(),
        }
      : undefined

  const agent: AtifAgent = {
    name: fork.agentName,
    version: '1.0.0',
    model_name: fork.modelId ? Option.some(fork.modelId) : Option.none(),
    tool_definitions: fork.agentRole ? Option.some(toolDefinitionsFromToolkit(fork.agentRole, configState, { solo })) : Option.none(),
    extra: Option.none(),
  }

  return {
    schema_version: 'ATIF-v1.7',
    session_id: sessionId ? Option.some(sessionId) : Option.none(),
    trajectory_id: Option.some(trajectoryId),
    agent,
    steps: fork.steps,
    notes: Option.none(),
    final_metrics: finalMetrics ? Option.some(finalMetrics) : Option.none(),
    continued_trajectory_ref: Option.none(),
    extra: Option.none(),
    subagent_trajectories: Option.none(),
  }
}

// =============================================================================
// Public API
// =============================================================================

export interface SerializeOptions {
  /** Tool definitions in OpenAI function-calling schema format (overrides auto-population) */
  toolDefinitions?: readonly Readonly<Record<string, JsonValue>>[]
  /** Session identifier */
  sessionId?: string
  /** Optional root-level notes */
  notes?: string
  /** Config state for vision-aware toolkit resolution */
  configState: ConfigState
  /** Solo mode — omits task/worker tools from serialized tool definitions */
  solo?: boolean
}

/**
 * Serialize the complete ATIF trajectory from fork states.
 *
 * The root fork (forkId=null) becomes the main trajectory. All other forks
 * are embedded as `subagent_trajectories[]`.
 */
export function serializeAtif(
  forks: ReadonlyMap<string | null, AtifForkState>,
  options: SerializeOptions,
): AtifTrajectory {
  const rootFork = forks.get(null)
  if (!rootFork) {
    throw new Error('[AtifProjection] No root fork found for serialization')
  }

  // Build subagent trajectories from worker forks
  const subagentTrajectories: AtifTrajectory[] = []
  for (const [forkId, fork] of forks.entries()) {
    if (forkId === null) continue
    // Use the forkId (agentId) as the trajectory_id for workers
    const traj = forkToTrajectory(fork, forkId, options.sessionId, options.configState, options.solo)
    subagentTrajectories.push(traj)
  }

  // Build root trajectory — aggregate sub-agent metrics into root total
  const rootMetrics = rootFork.tokenAccumulator
  const subMetric = (t: AtifTrajectory, field: 'total_prompt_tokens' | 'total_completion_tokens' | 'total_cached_tokens' | 'total_cost_usd' | 'total_steps'): number => {
    const fm = t.final_metrics
    if (!fm) return 0
    return Option.getOrElse(Option.flatMap(fm, m => m[field]), () => 0)
  }

  const totalPromptTokens = (rootMetrics.promptTokens ?? 0)
    + subagentTrajectories.reduce((sum, t) => sum + subMetric(t, 'total_prompt_tokens'), 0)
  const totalCompletionTokens = (rootMetrics.completionTokens ?? 0)
    + subagentTrajectories.reduce((sum, t) => sum + subMetric(t, 'total_completion_tokens'), 0)
  const totalCachedTokens = (rootMetrics.cachedTokens ?? 0)
    + subagentTrajectories.reduce((sum, t) => sum + subMetric(t, 'total_cached_tokens'), 0)
  const totalCostUsd = (rootMetrics.costUsd ?? 0)
    + subagentTrajectories.reduce((sum, t) => sum + subMetric(t, 'total_cost_usd'), 0)
  const totalSteps = rootFork.steps.length
    + subagentTrajectories.reduce((sum, t) => sum + subMetric(t, 'total_steps'), 0)

  const finalMetrics: AtifFinalMetrics | undefined =
    rootFork.steps.length > 0 || subagentTrajectories.length > 0
      ? {
          total_prompt_tokens: Option.some(totalPromptTokens),
          total_completion_tokens: Option.some(totalCompletionTokens),
          total_cached_tokens: Option.some(totalCachedTokens),
          total_cost_usd: Option.some(totalCostUsd),
          total_steps: Option.some(totalSteps),
          extra: Option.none(),
        }
      : undefined

  const agent: AtifAgent = {
    name: rootFork.agentName,
    version: '1.0.0',
    model_name: rootFork.modelId ? Option.some(rootFork.modelId) : Option.none(),
    tool_definitions: Option.fromNullable(options.toolDefinitions
      ?? (rootFork.agentRole ? toolDefinitionsFromToolkit(rootFork.agentRole, options.configState, { solo: options.solo }) : undefined)),
    extra: Option.none(),
  }

  return {
    schema_version: 'ATIF-v1.7',
    session_id: options.sessionId ? Option.some(options.sessionId) : Option.none(),
    trajectory_id: Option.some('main'),
    agent,
    steps: rootFork.steps,
    notes: options.notes ? Option.some(options.notes) : Option.none(),
    final_metrics: finalMetrics ? Option.some(finalMetrics) : Option.none(),
    continued_trajectory_ref: Option.none(),
    extra: Option.none(),
    subagent_trajectories: subagentTrajectories.length > 0 ? Option.some(subagentTrajectories) : Option.none(),
  }
}
