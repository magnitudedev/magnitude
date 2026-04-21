import GBNF from 'gbnf'
import { GrammarBuilder } from '../grammar-builder'
import type { GrammarToolDef } from '../grammar-builder'

/**
 * Sanitize grammar rule names for the `gbnf` npm library.
 * The library only allows lowercase letters and hyphens — no digits.
 * Replace digits in rule name identifiers with letter equivalents.
 */
export function sanitizeForGbnf(grammar: string): string {
  // Process line by line to distinguish rule names from quoted strings
  return grammar.split('\n').map(line => {
    const match = line.match(/^(\S+)(\s*::=\s*)(.+)$/)
    if (!match) return line

    const [, ruleName, sep, production] = match
    const sanitizedName = replaceDigits(ruleName)

    // In the production, replace rule name references but not quoted strings
    const sanitizedProduction = production.replace(
      /("(?:[^"\\]|\\.)*")|(\b[a-z][a-z0-9-]*\b)/g,
      (full, quoted, ident) => {
        if (quoted) return quoted // Don't touch quoted strings
        return replaceDigits(ident)
      }
    )

    return `${sanitizedName}${sep}${sanitizedProduction}`
  }).join('\n')
}

function replaceDigits(name: string): string {
  return name.replace(/[0-9]/g, d => String.fromCharCode('a'.charCodeAt(0) + parseInt(d)))
}

export interface GrammarValidator {
  /** Assert that the input is accepted by the grammar. Throws on rejection. */
  passes(input: string): void
  /** Assert that the input is rejected by the grammar. */
  rejects(input: string): void
  /** Get valid next characters/rules after feeding the input prefix. */
  validAfter(input: string): Array<{ type: string; value: number[] }>
  /** The sanitized grammar string */
  grammar: string
}

/**
 * Build a grammar validator from tool definitions and optional configuration.
 */
export function buildValidator(
  tools: GrammarToolDef[] = [],
  configure?: (b: GrammarBuilder) => GrammarBuilder
): GrammarValidator {
  let builder = GrammarBuilder.create(tools)
  if (configure) builder = configure(builder)
  const raw = builder.build()
  const sanitized = sanitizeForGbnf(raw)

  // Verify the grammar parses
  try {
    GBNF(sanitized)
  } catch (e: any) {
    throw new Error(`Grammar parse failed: ${e.message}\n\nGrammar:\n${sanitized}`)
  }

  return {
    passes(input: string): void {
      let state = GBNF(sanitized)
      for (let i = 0; i < input.length; i++) {
        try {
          state = state.add(input[i])
        } catch (e: any) {
          const ctx = input.slice(Math.max(0, i - 20), i)
          const after = input.slice(i + 1, i + 20)
          throw new Error(
            `Rejected at position ${i}, char ${JSON.stringify(input[i])}\n` +
            `Context: ...${JSON.stringify(ctx)} >>> ${JSON.stringify(input[i])} <<< ${JSON.stringify(after)}...\n` +
            `Original error: ${e.message}`
          )
        }
      }
    },

    rejects(input: string): void {
      let state = GBNF(sanitized)
      try {
        for (const ch of input) {
          state = state.add(ch)
        }
      } catch {
        return // Expected rejection
      }
      throw new Error(`Expected grammar to reject input but it was accepted: ${JSON.stringify(input.slice(0, 80))}`)
    },

    validAfter(input: string): Array<{ type: string; value: number[] }> {
      let state = GBNF(sanitized)
      for (const ch of input) {
        state = state.add(ch)
      }
      return [...state] as Array<{ type: string; value: number[] }>
    },

    grammar: sanitized,
  }
}

/** Standard shell tool with one scalar parameter */
export const SHELL_TOOL: GrammarToolDef = {
  tagName: 'shell',
  parameters: [{ name: 'command', field: 'command', type: 'scalar' }],
}

/** Skill tool with one scalar parameter */
export const SKILL_TOOL: GrammarToolDef = {
  tagName: 'skill',
  parameters: [{ name: 'name', field: 'name', type: 'scalar' }],
}

/** Tool with multiple parameters */
export const MULTI_PARAM_TOOL: GrammarToolDef = {
  tagName: 'edit',
  parameters: [
    { name: 'path', field: 'path', type: 'scalar' },
    { name: 'old', field: 'old', type: 'scalar' },
    { name: 'new', field: 'new', type: 'scalar' },
  ],
}

/** Tool with no parameters */
export const NO_PARAM_TOOL: GrammarToolDef = {
  tagName: 'tree',
  parameters: [],
}

/** Convenience: build validator with shell tool */
export function shellValidator(configure?: (b: GrammarBuilder) => GrammarBuilder) {
  return buildValidator([SHELL_TOOL], configure)
}

/** Convenience: build validator with multiple tools */
export function multiToolValidator(configure?: (b: GrammarBuilder) => GrammarBuilder) {
  return buildValidator([SHELL_TOOL, SKILL_TOOL, MULTI_PARAM_TOOL, NO_PARAM_TOOL], configure)
}
