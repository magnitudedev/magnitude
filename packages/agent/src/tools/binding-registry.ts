import type { XmlTagBinding } from '@magnitudedev/xml-act'
import type { RoleDefinition, ToolSet } from '@magnitudedev/roles'
import { defaultXmlTagName, getXmlBindingMap } from './index'

/**
 * Build a lightweight Map<tagName, XmlTagBinding> from an RoleDefinition.
 */
export function getBindingRegistry<TCtx>(
  agentDef: RoleDefinition<ToolSet, string, TCtx>,
): Map<string, XmlTagBinding> {
  const bindings = new Map<string, XmlTagBinding>()
  const xmlBindingMap = getXmlBindingMap()

  for (const [, tool] of Object.entries(agentDef.tools)) {
    if (!tool) continue
    const tagName = defaultXmlTagName(tool)
    const xmlBinding = xmlBindingMap.get(tagName)
    if (!xmlBinding) continue
    bindings.set(tagName, xmlBinding.toXmlTagBinding())
  }

  return bindings
}