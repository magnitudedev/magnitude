/**
 * XML Tool Documentation Generator
 *
 * Generates XML-ACT tool documentation from agent definitions.
 * Uses XML bindings on tools to produce tag documentation.
 *
 * Relocated from strategies/prompt-utils.ts after strategy abstraction removal.
 */

import type { Tool } from '@magnitudedev/tools'
import type { AgentDefinition, ToolSet } from '@magnitudedev/agent-definition'
import { generateXmlToolGroupDoc } from '@magnitudedev/xml-act'
import type { PolicyContext } from '../agents/types'
import { buildRegisteredTools } from './index'

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
  agentDef: AgentDefinition<ToolSet, PolicyContext>,
  implicitTools: readonly string[] = []
): ToolPresentation[] {
  const presentations: ToolPresentation[] = []

  for (const [defKey, tool] of Object.entries(agentDef.tools)) {
    if (implicitTools.includes(defKey)) continue
    if (!tool) continue

    const t = tool as Tool.Any
    const slug = agentDef.getSlug(defKey) ?? t.name

    presentations.push({
      slug,
      description: t.description,
      inputSchema: t.inputSchema,
      outputSchema: t.outputSchema,
      argMapping: t.argMapping ?? [],
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
  agentDef: AgentDefinition<ToolSet, PolicyContext>,
  implicitTools: readonly string[] = []
): string {
  // Build defKey lookup: tool instance → defKey (for implicit filtering)
  const defKeyLookup = new Map<Tool.Any, string>()
  for (const [defKey, tool] of Object.entries(agentDef.tools)) {
    if (tool) defKeyLookup.set(tool as Tool.Any, defKey)
  }

  // Group tools by group name for documentation
  const groups = new Map<string, { tools: Tool.Any[]; global: boolean }>()

  for (const tool of Object.values(agentDef.tools)) {
    if (!tool) continue
    const t = tool as Tool.Any
    const groupName = t.group ?? 'default'
    const isGlobal = !t.group || t.group === 'default'

    if (!groups.has(groupName)) {
      groups.set(groupName, { tools: [], global: isGlobal })
    }
    groups.get(groupName)!.tools.push(t)
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
