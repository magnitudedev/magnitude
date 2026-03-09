import type { Tool } from '@magnitudedev/tools'
import type { XmlTagBinding } from '@magnitudedev/xml-act'
import type { AgentDefinition, ToolSet } from '@magnitudedev/agent-definition'
function defaultXmlTagName(tool: Tool.Any): string {
  const group = tool.group
  if (!group || group === 'default') return tool.name
  return `${group}-${tool.name}`
}

/**
 * Build a lightweight Map<tagName, XmlTagBinding> from an AgentDefinition.
 */
export function getBindingRegistry(
  agentDef: AgentDefinition<ToolSet, unknown>,
): Map<string, XmlTagBinding> {
  const bindings = new Map<string, XmlTagBinding>()

  for (const [, tool] of Object.entries(agentDef.tools)) {
    if (!tool) continue
    const t = tool as Tool.Any
    const xmlInput = t.bindings?.xmlInput
    if (!xmlInput) continue

    const { type: _, ...binding } = xmlInput as { type: string } & XmlTagBinding
    const tagName = defaultXmlTagName(t)
    bindings.set(tagName, binding as XmlTagBinding)
  }

  return bindings
}