/**
 * XML Tool Documentation Generator
 *
 * Generates XML-ACT tool documentation from agent definitions.
 * Uses XML bindings on tools to produce tag documentation.
 *
 * Relocated from strategies/prompt-utils.ts after strategy abstraction removal.
 */

import type { RoleDefinition } from '@magnitudedev/roles'
import { generateXmlToolGroupDoc, generateXmlToolInputShape, type XmlToolDocEntry } from '@magnitudedev/xml-act'
import type { AgentCatalogEntry } from '../catalog'

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
  agentDef: RoleDefinition,
  implicitTools: readonly string[] = []
): ToolPresentation[] {
  const presentations: ToolPresentation[] = []

  for (const defKey of agentDef.tools.keys) {
    const entry = agentDef.tools.entries[defKey] as AgentCatalogEntry
    const tool = entry.tool
    if (implicitTools.includes(defKey)) continue

    const slug = defKey

    presentations.push({
      slug,
      description: typeof tool.description === 'string' ? tool.description : '',
      inputSchema: tool.inputSchema,
      outputSchema: tool.outputSchema,
      argMapping: [],

    })
  }

  return presentations
}

// =============================================================================
// XML-ACT Tool Documentation
// =============================================================================

function toXmlToolDocEntry(catalogEntry: AgentCatalogEntry): XmlToolDocEntry {
  const tool = catalogEntry.tool
  const xmlBinding = catalogEntry.binding

  return {
    name: tool.name,
    group: tool.group,
    description: tool.description,
    inputSchema: tool.inputSchema,
    outputSchema: tool.outputSchema,
    xmlInput: xmlBinding.toXmlTagBinding(),
  }
}

interface ResolvedXmlToolReference {
  readonly catalogEntry: AgentCatalogEntry
  readonly tagName: string
}

function generateXmlActToolShapeForEntry(catalogEntry: AgentCatalogEntry): string {
  return generateXmlToolInputShape(toXmlToolDocEntry(catalogEntry))
}

function resolveCatalogEntryForToolRef(
  agentDef: RoleDefinition,
  toolRef: string,
): ResolvedXmlToolReference | undefined {
  const directEntry = agentDef.tools.entries[toolRef] as AgentCatalogEntry | undefined
  if (directEntry) {
    return {
      catalogEntry: directEntry,
      tagName: directEntry.binding.toXmlTagBinding().tag,
    }
  }

  for (const defKey of agentDef.tools.keys) {
    const catalogEntry = agentDef.tools.entries[defKey] as AgentCatalogEntry
    const xmlInput = catalogEntry.binding.toXmlTagBinding()
    if (xmlInput.tag === toolRef || catalogEntry.tool.name === toolRef) {
      return {
        catalogEntry,
        tagName: xmlInput.tag,
      }
    }
  }

  return undefined
}

/**
 * Resolve a tool reference to the XML tag name the model should use.
 * Accepts the agent tool defKey first, then falls back to XML tag name or tool name.
 */
export function resolveXmlTagName(
  agentDef: RoleDefinition,
  toolRef: string,
): string | undefined {
  return resolveCatalogEntryForToolRef(agentDef, toolRef)?.tagName
}

/**
 * Generate the canonical XML input shape for a tool reference.
 * Accepts the agent tool defKey first, then falls back to XML tag name or tool name.
 */
export function generateXmlActToolShape(
  agentDef: RoleDefinition,
  toolRef: string,
): string | undefined {
  const resolved = resolveCatalogEntryForToolRef(agentDef, toolRef)
  return resolved ? generateXmlActToolShapeForEntry(resolved.catalogEntry) : undefined
}

/**
 * Generate XML-ACT tool documentation from an agent definition.
 * Uses XML bindings on tools to produce XML tag documentation grouped by namespace.
 */
export function generateXmlActToolDocs(
  agentDef: RoleDefinition,
  implicitTools: readonly string[] = []
): string {

  // Build defKey lookup: entry instance → defKey (for implicit filtering)
  const defKeyLookup = new Map<XmlToolDocEntry, string>()

  // Group tools by group name for documentation
  const groups = new Map<string, { tools: XmlToolDocEntry[]; global: boolean }>()

  for (const defKey of agentDef.tools.keys) {
    const catalogEntry = agentDef.tools.entries[defKey] as AgentCatalogEntry
    const entry = toXmlToolDocEntry(catalogEntry)

    defKeyLookup.set(entry, defKey)

    const groupName = entry.group ?? 'default'
    const isGlobal = !entry.group || entry.group === 'default'
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
