import {
  NativeChatCompletions,
  defaultClassifyStreamError,
  type ModelSpec,
  type StreamError,
} from "@magnitudedev/ai"
import { classifyMagnitudeConnectionError, type MagnitudeConnectionError } from "./errors"
import type { RoleId } from './contract'

/** Symmetric with MagnitudeConnectionError; extend when Magnitude-specific stream errors are needed. */
export type MagnitudeStreamError = StreamError

/** All models consumed by the agent must conform to this type. */
export type MagnitudeModelSpec = ModelSpec<{}, MagnitudeConnectionError, MagnitudeStreamError>

export interface MagnitudeCompatibleSpecConfig {
  modelId: string
  endpoint: string
  contextWindow: number
  maxOutputTokens: number
}

export function createMagnitudeCompatibleSpec(config: MagnitudeCompatibleSpecConfig): MagnitudeModelSpec {
  return NativeChatCompletions.model({
    modelId: config.modelId,
    endpoint: config.endpoint,
    contextWindow: config.contextWindow,
    maxOutputTokens: config.maxOutputTokens,
    options: {},
    classifyConnectionError: (failure) =>
      classifyMagnitudeConnectionError(failure),
    classifyStreamError: (failure) =>
      defaultClassifyStreamError(failure),
  })
}

export function createRoleSpec(roleId: RoleId, endpoint: string) {
  return NativeChatCompletions.model({
    modelId: `role/${roleId}`,
    endpoint,
    contextWindow: 0,
    maxOutputTokens: 0,
    options: {},
    classifyConnectionError: (failure) =>
      classifyMagnitudeConnectionError(failure),
    classifyStreamError: (failure) =>
      defaultClassifyStreamError(failure),
  })
}
