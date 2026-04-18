
/**
 * ResolvedToolSet — Single source of truth for tool availability per slot
 *
 * Replaces the `excludeTools` band-aid with a bundled type that ensures
 * all 4 tool consumers stay in sync.
 *
 * INVARIANT: The 4 consumers must stay in sync:
 * 1. Tool registry (buildRegisteredTools) — tools available at runtime
 * 2. XML grammar (generateToolGrammar) — tools the model can generate
 * 3. System prompt / tool docs (generateXmlActToolDocs) — tools described to model
 * 4. Binding registry (getBindingRegistry) — XML bindings for serialization
 *
 * If a tool is unavailable for a slot, it must be absent from all four.
 * If a tool is available, it must be present in all four.
 */

import type { RoleDefinition } from '@magnitudedev/roles'
import type { MagnitudeSlot } from '../model-slots'
import type { ConfigState } from '../ambient/config-ambient'

/**
 * ResolvedToolSet bundles all 4 tool representations for a specific slot.
 * Built once per turn. All consumers read from the same availableKeys.
 */
export interface ResolvedToolSet {
  readonly agentDef: RoleDefinition
  readonly availableKeys: ReadonlySet<string>  // filtered defKeys for this slot
  readonly slot: MagnitudeSlot
}

/**
 * Build a ResolvedToolSet for a slot.
 * Single decision site for tool availability.
 */
export function buildResolvedToolSet(
  agentDef: RoleDefinition,
  configState: ConfigState,
  slot: MagnitudeSlot,
): ResolvedToolSet {
  const slotConfig = configState.bySlot[slot]
  const isMagnitudeProvider = slotConfig.providerId === 'magnitude'
  const hasExaKey = !!process.env.EXA_API_KEY
  
  // Compute available keys by filtering agentDef.tools.keys
  const availableKeys = new Set<string>()
  for (const defKey of agentDef.tools.keys) {
    // webSearch excluded when no Magnitude provider AND no EXA key
    if (defKey === 'webSearch' && !isMagnitudeProvider && !hasExaKey) {
      continue
    }
    availableKeys.add(defKey)
  }

  return {
    agentDef,
    availableKeys,
    slot,
  }
}
