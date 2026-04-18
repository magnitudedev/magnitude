import type { XmlTagBinding } from '@magnitudedev/xml-act'
import type { AgentCatalogEntry } from '../catalog'
import type { ResolvedToolSet } from './resolved-toolset'

/**
 * Build a lightweight Map<tagName, XmlTagBinding> from a ResolvedToolSet.
 */
export function getBindingRegistry(
  toolSet: ResolvedToolSet,
): Map<string, XmlTagBinding> {
  const bindings = new Map<string, XmlTagBinding>()
  const agentDef = toolSet.agentDef

  for (const key of toolSet.availableKeys) {
    const entry = agentDef.tools.entries[key] as AgentCatalogEntry
    const binding = entry.binding.toXmlTagBinding()
    bindings.set(binding.tag, binding)
  }

  return bindings
}