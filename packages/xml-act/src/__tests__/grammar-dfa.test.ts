/**
 * Tests for DFA-based body content rules in the GBNF grammar generator.
 *
 * Includes:
 * 1. A GBNF DFA simulator
 * 2. Unit tests for generateBodyRules
 * 3. DFA correctness tests (accept/reject cases)
 * 4. Integration tests verifying grammar output
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { Effect } from 'effect'
import { generateBodyRules, generateGrammar, type GrammarToolDef } from '../grammar-generator'
import { defineXmlBinding } from '../xml-binding'
import { defineTool } from '@magnitudedev/tools'

// =============================================================================
// GBNF DFA Simulator
// =============================================================================

type GbnfTerm =
  | { type: 'literal'; value: string }
  | { type: 'charclass'; negate: boolean; chars: string }
  | { type: 'ref'; name: string }
  | { type: 'empty' }

type GbnfAlt = GbnfTerm[]
type GbnfRule = GbnfAlt[]

function parseGbnfRules(rulesText: string): Map<string, GbnfRule> {
  const map = new Map<string, GbnfRule>()
  for (const line of rulesText.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || !trimmed.includes('::=')) continue
    const idx = trimmed.indexOf('::=')
    const name = trimmed.slice(0, idx).trim()
    const rhs = trimmed.slice(idx + 3).trim()
    map.set(name, parseAlternatives(rhs))
  }
  return map
}

function parseAlternatives(rhs: string): GbnfRule {
  const alts: GbnfAlt[] = []
  let current: GbnfTerm[] = []
  let i = 0

  while (i < rhs.length) {
    if (rhs[i] === ' ' || rhs[i] === '\t') { i++; continue }

    if (rhs[i] === '|') {
      alts.push(current)
      current = []
      i++
      continue
    }

    if (rhs[i] === '"') {
      let val = ''
      i++
      while (i < rhs.length && rhs[i] !== '"') {
        if (rhs[i] === '\\' && i + 1 < rhs.length) {
          const next = rhs[i + 1]
          if (next === 'n') { val += '\n'; i += 2 }
          else if (next === 't') { val += '\t'; i += 2 }
          else if (next === '"') { val += '"'; i += 2 }
          else if (next === '\\') { val += '\\'; i += 2 }
          else { val += rhs[i]; i++ }
        } else {
          val += rhs[i]; i++
        }
      }
      i++
      if (val === '') {
        current.push({ type: 'empty' })
      } else {
        current.push({ type: 'literal', value: val })
      }
      continue
    }

    if (rhs[i] === '[') {
      i++
      let negate = false
      if (rhs[i] === '^') { negate = true; i++ }
      let chars = ''
      while (i < rhs.length && rhs[i] !== ']') {
        if (rhs[i] === '\\' && i + 1 < rhs.length) {
          chars += rhs[i + 1]; i += 2
        } else {
          chars += rhs[i]; i++
        }
      }
      i++
      current.push({ type: 'charclass', negate, chars })
      continue
    }

    let name = ''
    while (i < rhs.length && rhs[i] !== ' ' && rhs[i] !== '\t' && rhs[i] !== '|') {
      name += rhs[i]; i++
    }
    if (name) current.push({ type: 'ref', name })
  }

  alts.push(current)
  return alts
}

function matchRule(
  rules: Map<string, GbnfRule>,
  startRule: string,
  input: string,
): boolean {
  const memo = new Map<string, Set<number>>()

  function matchAt(ruleName: string, pos: number): Set<number> {
    const key = `${ruleName}@${pos}`
    if (memo.has(key)) return memo.get(key)!
    const result = new Set<number>()
    memo.set(key, result)
    const rule = rules.get(ruleName)
    if (!rule) return result
    for (const alt of rule) {
      for (const endPos of matchSeq(alt, pos)) {
        result.add(endPos)
      }
    }
    return result
  }

  function matchSeq(seq: GbnfTerm[], pos: number): Set<number> {
    if (seq.length === 0) return new Set([pos])
    const [head, ...tail] = seq
    const midPositions = matchTerm(head, pos)
    const results = new Set<number>()
    for (const mid of midPositions) {
      for (const end of matchSeq(tail, mid)) {
        results.add(end)
      }
    }
    return results
  }

  function matchTerm(term: GbnfTerm, pos: number): Set<number> {
    switch (term.type) {
      case 'empty': return new Set([pos])
      case 'literal': {
        if (input.startsWith(term.value, pos)) return new Set([pos + term.value.length])
        return new Set()
      }
      case 'charclass': {
        if (pos >= input.length) return new Set()
        const ch = input[pos]
        const inClass = term.chars.includes(ch)
        const matches = term.negate ? !inClass : inClass
        return matches ? new Set([pos + 1]) : new Set()
      }
      case 'ref': return matchAt(term.name, pos)
    }
  }

  return matchAt(startRule, 0).has(input.length)
}

function matchBody(prefix: string, tagName: string, body: string): boolean {
  const rulesText = generateBodyRules(prefix, tagName).join('\n')
  const rules = parseGbnfRules(rulesText)
  return matchRule(rules, `${prefix}-body`, body)
}

// =============================================================================
// Unit tests for generateBodyRules
// =============================================================================

describe('generateBodyRules', () => {
  it('generates a body entry rule and state rules', () => {
    const rules = generateBodyRules('writetool', 'write')
    const text = rules.join('\n')
    expect(text).toContain('writetool-body ::= writetool-body-s0')
    expect(text).toContain('writetool-body-s0 ::=')
    expect(text).toContain('writetool-body-s1 ::=')
  })

  it('generates correct number of rules for tag write', () => {
    // closing tag </write> has 8 chars, generates states s0..s7 = 8 state rules + 1 entry = 9
    const rules = generateBodyRules('writetool', 'write')
    expect(rules).toHaveLength(9)
  })

  it('uses the correct prefix in all rule names', () => {
    const rules = generateBodyRules('shelltool', 'shell')
    for (const rule of rules) {
      expect(rule.startsWith('shelltool-body')).toBe(true)
    }
  })

  it('works for single-char tag names', () => {
    // closing tag </a> has 4 chars, generates states s0..s3 = 4 state rules + 1 entry = 5
    const rules = generateBodyRules('p', 'a')
    expect(rules).toHaveLength(5)
  })

  it('s0 rule allows empty match', () => {
    const rules = generateBodyRules('writetool', 'write')
    const s0 = rules.find(r => r.startsWith('writetool-body-s0'))!
    expect(s0).toContain('""')
  })
})

// =============================================================================
// DFA correctness tests - write tag
// =============================================================================

describe('DFA body matcher - write tag', () => {
  const tag = 'write'
  const prefix = 'writetool'
  const CLOSING = '</write>'

  it('accepts empty body', () => {
    expect(matchBody(prefix, tag, '')).toBe(true)
  })

  it('accepts plain text', () => {
    expect(matchBody(prefix, tag, 'hello world')).toBe(true)
  })

  it('accepts body with generic type syntax', () => {
    expect(matchBody(prefix, tag, 'Array<number>')).toBe(true)
  })

  it('accepts body with multiple angle brackets', () => {
    expect(matchBody(prefix, tag, 'Map<string, Array<number>>')).toBe(true)
  })

  it('accepts body with </ not followed by tagname', () => {
    expect(matchBody(prefix, tag, 'foo</div>bar')).toBe(true)
  })

  it('accepts body with partial closing tag prefix', () => {
    expect(matchBody(prefix, tag, 'foo</writ')).toBe(true)
  })

  it('accepts body with </write without closing >', () => {
    expect(matchBody(prefix, tag, 'foo</write')).toBe(true)
  })

  it('accepts comparison operators', () => {
    expect(matchBody(prefix, tag, 'if (a < b && c > d) return true')).toBe(true)
  })

  it('accepts body with HTML-like content', () => {
    expect(matchBody(prefix, tag, '<div>hello</span>')).toBe(true)
  })

  it('accepts body with newlines and tabs', () => {
    expect(matchBody(prefix, tag, 'line1\nline2\ttabbed')).toBe(true)
  })

  it('accepts </writes> (extra char after tagname)', () => {
    expect(matchBody(prefix, tag, '</writes>')).toBe(true)
  })

  it('rejects body containing the exact closing tag', () => {
    expect(matchBody(prefix, tag, 'foo' + CLOSING + 'bar')).toBe(false)
  })

  it('rejects body that IS the closing tag', () => {
    expect(matchBody(prefix, tag, CLOSING)).toBe(false)
  })

  it('rejects body ending with the closing tag', () => {
    expect(matchBody(prefix, tag, 'some content' + CLOSING)).toBe(false)
  })

  it('rejects body starting with the closing tag', () => {
    expect(matchBody(prefix, tag, CLOSING + 'trailing')).toBe(false)
  })
})

// =============================================================================
// DFA correctness tests - shell tag
// =============================================================================

describe('DFA body matcher - shell tag', () => {
  const tag = 'shell'
  const prefix = 'shelltool'
  const CLOSING = '</shell>'

  it('accepts shell command with redirects', () => {
    expect(matchBody(prefix, tag, 'cat file.txt > output.txt')).toBe(true)
  })

  it('accepts shell command with angle bracket syntax', () => {
    expect(matchBody(prefix, tag, 'echo hello | grep <pattern>')).toBe(true)
  })

  it('rejects body containing exact closing tag', () => {
    expect(matchBody(prefix, tag, 'echo foo' + CLOSING + 'echo bar')).toBe(false)
  })

  it('accepts closing tag of different element inside shell body', () => {
    expect(matchBody(prefix, tag, 'echo </write>')).toBe(true)
  })
})

// =============================================================================
// DFA correctness tests - message tag
// =============================================================================

describe('DFA body matcher - message tag', () => {
  const tag = 'message'
  const prefix = 'msg'
  const CLOSING = '</message>'

  it('accepts message with code snippet containing generics', () => {
    expect(matchBody(prefix, tag, 'Use Array<string> for this')).toBe(true)
  })

  it('accepts message with HTML tags', () => {
    expect(matchBody(prefix, tag, 'See <b>bold</b> text')).toBe(true)
  })

  it('accepts empty message', () => {
    expect(matchBody(prefix, tag, '')).toBe(true)
  })

  it('rejects body containing exact closing tag', () => {
    expect(matchBody(prefix, tag, 'foo' + CLOSING + 'bar')).toBe(false)
  })
})

// =============================================================================
// DFA correctness tests - lens tag
// =============================================================================

describe('DFA body matcher - lens tag', () => {
  const tag = 'lens'
  const prefix = 'lens'
  const CLOSING = '</lens>'

  it('accepts lens content with angle brackets', () => {
    expect(matchBody(prefix, tag, 'Array<T> type handling needed')).toBe(true)
  })

  it('accepts empty lens body', () => {
    expect(matchBody(prefix, tag, '')).toBe(true)
  })

  it('rejects body containing exact closing tag', () => {
    expect(matchBody(prefix, tag, 'foo' + CLOSING + 'bar')).toBe(false)
  })
})

// =============================================================================
// DFA correctness tests - edge cases
// =============================================================================

describe('DFA body matcher - edge cases', () => {
  it('handles tag name with repeated characters', () => {
    expect(matchBody('p', 'aaa', 'foo</aa>bar')).toBe(true)
    expect(matchBody('p', 'aaa', 'foo</aaa>bar')).toBe(false)
  })

  it('handles single-char tag name', () => {
    expect(matchBody('p', 'a', 'hello')).toBe(true)
    expect(matchBody('p', 'a', '</a>')).toBe(false)
    expect(matchBody('p', 'a', 'x</a>y')).toBe(false)
    expect(matchBody('p', 'a', '</b>')).toBe(true)
  })

  it('accepts content that looks like closing tag but has extra chars', () => {
    expect(matchBody('writetool', 'write', '</writes>')).toBe(true)
  })

  it('accepts multiple consecutive < characters', () => {
    expect(matchBody('writetool', 'write', '<<<')).toBe(true)
  })

  it('accepts content with mixed special chars', () => {
    expect(matchBody('writetool', 'write', '<>/<>/><')).toBe(true)
  })

  it('rejects closing tag in the middle of longer content', () => {
    expect(matchBody('writetool', 'write', 'before</write>after')).toBe(false)
  })
})

// =============================================================================
// Integration tests - grammar output
// =============================================================================

const writeTool = defineTool({
  name: 'write',
  group: 'fs',
  description: 'Write file',
  inputSchema: Schema.Struct({ path: Schema.String, content: Schema.String }),
  outputSchema: Schema.Void,
  execute: () => Effect.succeed(undefined),
})

const writeBinding = defineXmlBinding(writeTool, {
  input: {
    attributes: [{ field: 'path', attr: 'path' }],
    body: 'content',
  },
  output: {},
} as const)

const editTool = defineTool({
  name: 'edit',
  group: 'fs',
  description: 'Edit file',
  inputSchema: Schema.Struct({
    path: Schema.String,
    oldString: Schema.String,
    newString: Schema.String,
  }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})

const editBinding = defineXmlBinding(editTool, {
  input: {
    attributes: [{ field: 'path', attr: 'path' }],
    childTags: [
      { tag: 'old', field: 'oldString' },
      { tag: 'new', field: 'newString' },
    ],
  },
  output: {},
} as const)

const readTool = defineTool({
  name: 'read',
  group: 'fs',
  description: 'Read file',
  inputSchema: Schema.Struct({ path: Schema.String }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})

const readBinding = defineXmlBinding(readTool, {
  input: { attributes: [{ field: 'path', attr: 'path' }] },
  output: {},
} as const)

function makeDef(binding: any, tool: any): GrammarToolDef {
  const tagBinding = binding.toXmlTagBinding()
  return { tagName: tagBinding.tag, binding: tagBinding, inputSchema: tool.inputSchema }
}

describe('generateGrammar integration - DFA body rules', () => {
  it('lens rule uses DFA body', () => {
    const grammar = generateGrammar([])
    expect(grammar).toContain('lens-body')
    expect(grammar).toContain('lens-body-s0')
  })

  it('message rule uses DFA body', () => {
    const grammar = generateGrammar([])
    expect(grammar).toContain('msg-body')
    expect(grammar).toContain('msg-body-s0')
  })

  it('tool with body uses DFA body rules', () => {
    const grammar = generateGrammar([makeDef(writeBinding, writeTool)])
    expect(grammar).toContain('writetool-body')
    expect(grammar).toContain('writetool-body-s0')
    expect(grammar).toMatch(/writetool ::=.*writetool-body/)
  })

  it('child tags with body use DFA body rules', () => {
    const grammar = generateGrammar([makeDef(editBinding, editTool)])
    expect(grammar).toContain('edittool-oldtool-body')
    expect(grammar).toContain('edittool-newtool-body')
  })

  it('self-closing tool does not generate body rules', () => {
    const grammar = generateGrammar([makeDef(readBinding, readTool)])
    expect(grammar).not.toContain('readtool-body')
  })

  it('grammar contains no [^<]+ patterns', () => {
    const grammar = generateGrammar([
      makeDef(writeBinding, writeTool),
      makeDef(editBinding, editTool),
      makeDef(readBinding, readTool),
    ])
    expect(grammar).not.toContain('[^<]+')
  })

  it('body entry rule references s0', () => {
    const grammar = generateGrammar([makeDef(writeBinding, writeTool)])
    expect(grammar).toContain('writetool-body ::= writetool-body-s0')
  })
})
