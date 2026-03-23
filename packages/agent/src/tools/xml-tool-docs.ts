/**
 * XML Tool Documentation Generator
 *
 * Generates XML-ACT tool documentation from agent definitions.
 * Uses XML bindings on tools to produce tag documentation.
 *
 * Relocated from strategies/prompt-utils.ts after strategy abstraction removal.
 */

import type { RoleDefinition, ToolSet } from '@magnitudedev/roles'
import { generateXmlToolGroupDoc, type XmlToolDocEntry } from '@magnitudedev/xml-act'
import type { PolicyContext } from '../agents/types'
import { defaultXmlTagName, getXmlBindingMap } from './index'

// =============================================================================
// Tool Presentation (strategy-agnostic tool metadata)
// =============================================================================

export interface ToolPresentation {
  /** Slug as the model should reference it (e.g., 'fs.read', 'shell') */
  readonly slug: string
  /** Human-readable description */
  readonly description: string
  /** Input schema for auto-generating type signatures */
  readonly inputSchema: unknown
  /** Output schema for auto-generating return types */
  readonly outputSchema: unknown
  /** Positional argument mapping (ordered names for call syntax) */
  readonly argMapping: readonly string[]
}

/**
 * Build ToolPresentation[] from an agent definition.
 */
export function buildToolPresentation(
  agentDef: RoleDefinition<ToolSet, string, PolicyContext>,
  implicitTools: readonly string[] = []
): ToolPresentation[] {
  const presentations: ToolPresentation[] = []

  for (const [defKey, tool] of Object.entries(agentDef.tools)) {
    if (implicitTools.includes(defKey)) continue
    if (!tool) continue

    const slug = defKey

    presentations.push({
      slug,
      description: 'description' in tool && typeof tool.description === 'string' ? tool.description : '',
      inputSchema: tool.inputSchema,
      outputSchema: tool.outputSchema,
      argMapping: 'argMapping' in tool && Array.isArray(tool.argMapping) ? tool.argMapping : [],
    })
  }

  return presentations
}

// =============================================================================
// XML-ACT Tool Documentation
// =============================================================================

/**
 * Generate XML-ACT tool documentation from an agent definition.
 * Uses XML bindings on tools to produce XML tag documentation grouped by namespace.
 */
export function generateXmlActToolDocs(
  agentDef: RoleDefinition<ToolSet, string, PolicyContext>,
  implicitTools: readonly string[] = []
): string {
  const xmlBindingMap = getXmlBindingMap()

  // Build defKey lookup: entry instance → defKey (for implicit filtering)
  const defKeyLookup = new Map<XmlToolDocEntry, string>()

  // Group tools by group name for documentation
  const groups = new Map<string, { tools: XmlToolDocEntry[]; global: boolean }>()

  for (const [defKey, tool] of Object.entries(agentDef.tools)) {
    if (!tool) continue

    const tagName = defaultXmlTagName(tool)
    const xmlBinding = xmlBindingMap.get(tagName)
    if (!xmlBinding) continue

    const entry: XmlToolDocEntry = {
      name: tool.name,
      group: tool.group,
      description: tool.description,
      inputSchema: tool.inputSchema,
      outputSchema: tool.outputSchema,
      xmlInput: xmlBinding.toXmlTagBinding(),
      xmlOutput: xmlBinding.toXmlOutputBinding(),
    }

    defKeyLookup.set(entry, defKey)

    const groupName = tool.group ?? 'default'
    const isGlobal = !tool.group || tool.group === 'default'
    if (!groups.has(groupName)) groups.set(groupName, { tools: [], global: isGlobal })
    groups.get(groupName)!.tools.push(entry)
  }

  const parts: string[] = []

  for (const [groupName, group] of groups) {
    const doc = generateXmlToolGroupDoc(
      group.global ? 'Global' : groupName,
      group.tools,
      implicitTools,
      defKeyLookup,
    )

    if (doc) parts.push(doc)
  }

  return parts.join('\n\n')
}
