/**
 * Frontmatter Utility
 * 
 * Shared utility for parsing and serializing markdown files with YAML frontmatter.
 * Used by skill scanning and spec persistence.
 */

export interface FrontmatterResult<T = Record<string, any>> {
  frontmatter: T
  body: string
}

/**
 * Parse markdown with YAML frontmatter.
 * 
 * Expects frontmatter delimited by --- at the start of the content.
 * Uses Bun.YAML.parse for robust YAML parsing.
 * 
 * @param content - Markdown content with frontmatter
 * @returns Parsed frontmatter and body, or null if no frontmatter found
 * 
 * @example
 * ```typescript
 * const result = parseFrontmatter(`---
 * name: my-skill
 * description: Does cool things
 * ---
 * 
 * # Content here
 * `)
 * // { frontmatter: { name: 'my-skill', description: 'Does cool things' }, body: '# Content here' }
 * ```
 */
export function parseFrontmatter<T = Record<string, any>>(
  content: string
): FrontmatterResult<T> | null {
  // Match frontmatter between --- delimiters at start of content
  // Support both \n and \r\n line endings
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!match) {
    return null
  }

  const frontmatterYaml = match[1]
  const body = content.slice(match[0].length).trim()

  try {
    // Use Bun.YAML.parse for robust YAML parsing (handles both block and flow styles)
    const frontmatter = Bun.YAML.parse(frontmatterYaml) as T
    return { frontmatter, body }
  } catch (error) {
    // Invalid YAML - return null
    return null
  }
}

/**
 * Serialize frontmatter and body to markdown.
 * 
 * Formats frontmatter as block-style YAML for readability.
 * 
 * @param frontmatter - Object to serialize as YAML frontmatter
 * @param body - Markdown body content
 * @returns Complete markdown with frontmatter
 * 
 * @example
 * ```typescript
 * const md = serializeFrontmatter(
 *   { name: 'my-skill', version: '1.0' },
 *   '# My Skill\n\nContent here'
 * )
 * // ---
 * // name: my-skill
 * // version: 1.0
 * // ---
 * // 
 * // # My Skill
 * // 
 * // Content here
 * ```
 */
export function serializeFrontmatter(
  frontmatter: Record<string, any>,
  body: string
): string {
  const frontmatterBlock = Bun.YAML.stringify(frontmatter, null, 2).trimEnd()
  return `---\n${frontmatterBlock}\n---\n\n${body}`
}
