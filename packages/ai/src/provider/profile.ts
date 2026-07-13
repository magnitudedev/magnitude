import type { ProviderModelCapabilities } from "../model/capabilities"
import type { ProviderModel } from "./model"

/**
 * Model metadata needed by the agent runtime — a strict subset of
 * ProviderModel. Covers context limits, output defaults, and capability
 * flags.
 */
export interface ModelProfile {
  readonly contextWindow: number
  readonly maxOutputTokens: number
  readonly capabilities: ProviderModelCapabilities
}

/**
 * Extract a ModelProfile from a ProviderModel.
 *
 * Works on any ProviderModel (including subtypes like MagnitudeModelInfo).
 */
export function toModelProfile(model: ProviderModel): ModelProfile {
  return {
    contextWindow: model.contextWindow,
    maxOutputTokens: model.maxOutputTokens,
    capabilities: model.capabilities,
  }
}
