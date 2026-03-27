import type { XmlTagBinding } from '@magnitudedev/xml-act'
import type { RoleDefinition } from '@magnitudedev/roles'
import type { AgentCatalogEntry } from '../catalog'

/**
 * Build a lightweight Map<tagName, XmlTagBinding> from an RoleDefinition.
 */
export function getBindingRegistry(
  agentDef: RoleDefinition,
): Map<string, XmlTagBinding> {
  const bindings = new Map<string, XmlTagBinding>()

  for (const key of agentDef.tools.keys) {
    const entry = agentDef.tools.entries[key] as AgentCatalogEntry
    const binding = entry.binding.toXmlTagBinding()
    bindings.set(binding.tag, binding)
  }

  return bindings
}