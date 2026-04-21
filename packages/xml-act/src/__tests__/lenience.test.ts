/**
 * Comprehensive leniency test suite for xml-act tokenizer and grammar.
 *
 * Covers all lenience mechanisms documented in packages/xml-act/docs/lenience.md:
 *   - Closing tag Modes 1-3 (slash-prefix, pipe-omission)
 *   - Opening tag invoke-without-keyword
 *   - Newline enforcement (strictNewlines)
 *   - Grammar DFA lenient close variants
 *   - Grammar newline embedding
 */

import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'
import { generateBodyRules, buildToolRules, GrammarBuilder } from '../grammar/grammar-builder'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DEFAULT_OPTIONS = { strictNewlines: true, toolKeyword: 'invoke' }

function collect(
  input: string | string[],
  knownToolTags: ReadonlySet<string> = new Set(),
  options: { strictNewlines: boolean; toolKeyword: string } = DEFAULT_OPTIONS,
): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer(
    (token) => {
      const { raw, ...rest } = token as any
      out.push(rest)
    },
    knownToolTags,
    options,
  )
  if (Array.isArray(input)) {
    for (const chunk of input) tokenizer.push(chunk)
  } else {
    tokenizer.push(input)
  }
  tokenizer.end()
  return out
}

function collectWithTools(input: string | string[], tools: string[]): any[] {
  return collect(input, new Set(tools))
}

// ---------------------------------------------------------------------------
// 1. Closing tag lenience — tokenizer
// ---------------------------------------------------------------------------

describe('Closing tag leniency (tokenizer)', () => {
  describe('Mode 1: </name|> — slash-prefix, pipe retained', () => {
    it('accepts </think|>', () => {
      expect(collect('</think|>')).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('accepts </message|>', () => {
      expect(collect('</message|>')).toEqual([{ _tag: 'Close', name: 'message', pipe: undefined }])
    })

    it('accepts </invoke|>', () => {
      expect(collect('</invoke|>')).toEqual([{ _tag: 'Close', name: 'invoke', pipe: undefined }])
    })

    it('accepts </yield|>', () => {
      expect(collect('</yield|>')).toEqual([{ _tag: 'Close', name: 'yield', pipe: undefined }])
    })

    it('accepts </parameter|>', () => {
      expect(collect('</parameter|>')).toEqual([{ _tag: 'Close', name: 'parameter', pipe: undefined }])
    })

    it('normalizes to canonical Close token (no slash in output)', () => {
      const tokens = collect('</think|>')
      expect(tokens[0]._tag).toBe('Close')
      expect(tokens[0].name).toBe('think')
    })
  })

  describe('Mode 2: </name> — slash-prefix, pipe omitted', () => {
    it('accepts </think>', () => {
      expect(collect('</think>')).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('accepts </message>', () => {
      expect(collect('</message>')).toEqual([{ _tag: 'Close', name: 'message', pipe: undefined }])
    })

    it('accepts </invoke>', () => {
      expect(collect('</invoke>')).toEqual([{ _tag: 'Close', name: 'invoke', pipe: undefined }])
    })

    it('accepts </yield>', () => {
      expect(collect('</yield>')).toEqual([{ _tag: 'Close', name: 'yield', pipe: undefined }])
    })

    it('accepts </parameter>', () => {
      expect(collect('</parameter>')).toEqual([{ _tag: 'Close', name: 'parameter', pipe: undefined }])
    })
  })

  describe('Mode 3: <name> — no slash, no pipe', () => {
    it('accepts <think> as close', () => {
      expect(collect('<think>')).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('accepts <message> as close', () => {
      expect(collect('<message>')).toEqual([{ _tag: 'Close', name: 'message', pipe: undefined }])
    })

    it('accepts <invoke> as close', () => {
      expect(collect('<invoke>')).toEqual([{ _tag: 'Close', name: 'invoke', pipe: undefined }])
    })

    it('accepts <yield> as close', () => {
      expect(collect('<yield>')).toEqual([{ _tag: 'Close', name: 'yield', pipe: undefined }])
    })

    it('accepts <parameter> as close', () => {
      expect(collect('<parameter>')).toEqual([{ _tag: 'Close', name: 'parameter', pipe: undefined }])
    })
  })

  describe('Canonical form still works', () => {
    it('accepts <think|>', () => {
      expect(collect('<think|>')).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('accepts <message|>', () => {
      expect(collect('<message|>')).toEqual([{ _tag: 'Close', name: 'message', pipe: undefined }])
    })
  })

  describe('Chunk boundary cases', () => {
    it('handles </ split across chunks: ["<", "/think|>"]', () => {
      expect(collect(['<', '/think|>'])).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('handles </ split across chunks: ["</", "think|>"]', () => {
      expect(collect(['</', 'think|>'])).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('handles </think split: ["</thi", "nk|>"]', () => {
      expect(collect(['</thi', 'nk|>'])).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('handles </think> split: ["<", "/think>"]', () => {
      expect(collect(['<', '/think>'])).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('handles </think> split: ["</think", ">"]', () => {
      expect(collect(['</think', '>'])).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('handles </think|> split at pipe: ["</think", "|>"]', () => {
      expect(collect(['</think', '|>'])).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('handles char-by-char </think|>', () => {
      const chars = '</think|>'.split('')
      expect(collect(chars)).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('handles char-by-char </think>', () => {
      const chars = '</think>'.split('')
      expect(collect(chars)).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })
  })

  describe('Invalid names after </ — failAsContent', () => {
    it('treats </ > as content (space after slash)', () => {
      const tokens = collect('</ >')
      expect(tokens).toEqual([{ _tag: 'Content', text: '</ >' }])
    })

    it('treats </> as content (no name)', () => {
      const tokens = collect('</>')
      // < starts close, / is skipped, > is invalid first char → failAsContent
      expect(tokens).toEqual([{ _tag: 'Content', text: '</>' }])
    })

    it('treats </123> as content (digit first char)', () => {
      const tokens = collect('</123>')
      expect(tokens).toEqual([{ _tag: 'Content', text: '</123>' }])
    })
  })

  describe('Mixed: canonical open + lenient close', () => {
    it('canonical open + Mode 1 close', () => {
      const tokens = collect('<|think:strategy>body\n</think|>')
      expect(tokens).toEqual([
        { _tag: 'Open', name: 'think', variant: 'strategy' },
        { _tag: 'Content', text: 'body\n' },
        { _tag: 'Close', name: 'think', pipe: undefined },
      ])
    })

    it('canonical open + Mode 2 close', () => {
      const tokens = collect('<|message:user>hello\n</message>')
      expect(tokens).toEqual([
        { _tag: 'Open', name: 'message', variant: 'user' },
        { _tag: 'Content', text: 'hello\n' },
        { _tag: 'Close', name: 'message', pipe: undefined },
      ])
    })

    it('canonical open + Mode 3 close', () => {
      const tokens = collect('<|message:user>hello\n<message>')
      expect(tokens).toEqual([
        { _tag: 'Open', name: 'message', variant: 'user' },
        { _tag: 'Content', text: 'hello\n' },
        { _tag: 'Close', name: 'message', pipe: undefined },
      ])
    })

    it('full turn with lenient closes', () => {
      const input = '<|think:strategy>reasoning\n</think>\n<|message:user>hi\n</message>'
      const tokens = collect(input)
      expect(tokens.filter(t => t._tag === 'Open')).toHaveLength(2)
      expect(tokens.filter(t => t._tag === 'Close')).toHaveLength(2)
      expect(tokens.every(t => t._tag !== 'Content' || !t.text.includes('</'))).toBe(true)
    })
  })
})

// ---------------------------------------------------------------------------
// 2. Opening tag leniency — invoke-without-keyword
// ---------------------------------------------------------------------------

describe('Opening tag leniency — invoke-without-keyword (tokenizer)', () => {
  it('rewrites <|shell> to invoke:shell when shell is in knownToolTags', () => {
    expect(collectWithTools('<|shell>', ['shell'])).toEqual([
      { _tag: 'Open', name: 'invoke', variant: 'shell' },
    ])
  })

  it('rewrites <|read> to invoke:read', () => {
    expect(collectWithTools('<|read>', ['read', 'write'])).toEqual([
      { _tag: 'Open', name: 'invoke', variant: 'read' },
    ])
  })

  it('rewrites <|spawn-worker> to invoke:spawn-worker', () => {
    expect(collectWithTools('<|spawn-worker>', ['spawn-worker'])).toEqual([
      { _tag: 'Open', name: 'invoke', variant: 'spawn-worker' },
    ])
  })

  it('does NOT rewrite unknown tag not in knownToolTags', () => {
    expect(collectWithTools('<|unknown>', ['shell'])).toEqual([
      { _tag: 'Open', name: 'unknown', variant: undefined },
    ])
  })

  it('does NOT rewrite when variant is already present', () => {
    expect(collectWithTools('<|invoke:shell>', ['shell'])).toEqual([
      { _tag: 'Open', name: 'invoke', variant: 'shell' },
    ])
  })

  it('does NOT rewrite when knownToolTags is empty', () => {
    expect(collect('<|shell>', new Set())).toEqual([
      { _tag: 'Open', name: 'shell', variant: undefined },
    ])
  })

  it('does NOT rewrite when knownToolTags is undefined', () => {
    expect(collect('<|shell>')).toEqual([
      { _tag: 'Open', name: 'shell', variant: undefined },
    ])
  })

  it('uses custom toolKeyword', () => {
    const out = collect('<|shell>', new Set(['shell']), { toolKeyword: 'tool' })
    expect(out).toEqual([{ _tag: 'Open', name: 'tool', variant: 'shell' }])
  })

  it('rewrites across chunk boundary: ["<|", "shell>"]', () => {
    expect(collectWithTools(['<|', 'shell>'], ['shell'])).toEqual([
      { _tag: 'Open', name: 'invoke', variant: 'shell' },
    ])
  })

  it('rewrites across chunk boundary: ["<|she", "ll>"]', () => {
    expect(collectWithTools(['<|she', 'll>'], ['shell'])).toEqual([
      { _tag: 'Open', name: 'invoke', variant: 'shell' },
    ])
  })
})

// ---------------------------------------------------------------------------
// 3. Newline enforcement — strictNewlines
// ---------------------------------------------------------------------------

describe('Newline enforcement (tokenizer)', () => {
  describe('top-level open tags', () => {
    it('accepts <|think> preceded by newline', () => {
      const tokens = collect('\n<|think>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'think')).toBe(true)
    })

    it('rejects <|think> NOT preceded by newline', () => {
      const tokens = collect('text<|think>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'think')).toBe(false)
      const content = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
      expect(content).toContain('<|think>')
    })

    it('accepts <|message:user> preceded by newline', () => {
      const tokens = collect('\n<|message:user>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'message' && t.variant === 'user')).toBe(true)
    })

    it('rejects <|message:user> NOT preceded by newline', () => {
      const tokens = collect('text<|message:user>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'message')).toBe(false)
    })

    it('accepts <|invoke:shell> preceded by newline', () => {
      const tokens = collect('\n<|invoke:shell>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'invoke')).toBe(true)
    })

    it('rejects <|invoke:shell> NOT preceded by newline', () => {
      const tokens = collect('text<|invoke:shell>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'invoke')).toBe(false)
    })

    it('accepts <|yield:user|> preceded by newline', () => {
      const tokens = collect('\n<|yield:user|>')
      expect(tokens.some(t => t._tag === 'SelfClose' && t.name === 'yield')).toBe(true)
    })

    it('rejects <|yield:user|> NOT preceded by newline', () => {
      const tokens = collect('text<|yield:user|>')
      expect(tokens.some(t => t._tag === 'SelfClose' && t.name === 'yield')).toBe(false)
    })
  })

  describe('top-level close tags', () => {
    it('accepts <think|> preceded by newline', () => {
      const tokens = collect('\n<think|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(true)
    })

    it('rejects <think|> NOT preceded by newline', () => {
      const tokens = collect('text<think|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(false)
    })

    it('accepts <message|> preceded by newline', () => {
      const tokens = collect('\n<message|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'message')).toBe(true)
    })

    it('rejects <message|> NOT preceded by newline', () => {
      const tokens = collect('text<message|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'message')).toBe(false)
    })

    it('accepts <invoke|> preceded by newline', () => {
      const tokens = collect('\n<invoke|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'invoke')).toBe(true)
    })
  })

  describe('parameter tags exempt', () => {
    it('accepts <|parameter:foo> without preceding newline', () => {
      const tokens = collect('text<|parameter:foo>')
      expect(tokens.some(t => t._tag === 'Parameter' && t.name === 'foo')).toBe(true)
    })

    it('accepts <parameter|> without preceding newline', () => {
      const tokens = collect('text<parameter|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'parameter')).toBe(true)
    })
  })

  describe('start of stream counts as after newline', () => {
    it('accepts <|think> at start of stream (no preceding content)', () => {
      const tokens = collect('<|think>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'think')).toBe(true)
    })

    it('accepts <think|> at start of stream', () => {
      const tokens = collect('<think|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(true)
    })
  })

  describe('horizontal whitespace after newline (\\n + spaces/tabs + tag)', () => {
    it('accepts open tag with spaces after newline', () => {
      const tokens = collect('\n  <|message:user>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'message' && t.variant === 'user')).toBe(true)
    })

    it('accepts open tag with tab after newline', () => {
      const tokens = collect('\n\t<|think:strategy>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'think')).toBe(true)
    })

    it('accepts open tag with mixed spaces and tabs after newline', () => {
      const tokens = collect('\n \t <|invoke:shell>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'invoke')).toBe(true)
    })

    it('accepts close tag with spaces after newline', () => {
      const tokens = collect('\n  <think|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(true)
    })

    it('accepts close tag with tab after newline', () => {
      const tokens = collect('\n\t<message|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'message')).toBe(true)
    })

    it('accepts self-close tag with spaces after newline', () => {
      const tokens = collect('\n  <|yield:user|>')
      expect(tokens.some(t => t._tag === 'SelfClose' && t.name === 'yield')).toBe(true)
    })

    it('accepts lenient close (Mode 1) with spaces after newline', () => {
      const tokens = collect('\n  </think|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(true)
    })

    it('accepts lenient close (Mode 2) with spaces after newline', () => {
      const tokens = collect('\n  </message>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'message')).toBe(true)
    })

    it('accepts lenient close (Mode 3) with tab after newline', () => {
      const tokens = collect('\n\t<invoke>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'invoke')).toBe(true)
    })

    it('rejects tag with only spaces (no preceding newline)', () => {
      const tokens = collect('text  <|think:strategy>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'think')).toBe(false)
    })

    it('accepts full turn with indented tags', () => {
      const input = ' <|think:strategy>reasoning\n <think|>\n <|message:user>\nHey! How can I help?\n <message|>\n <|yield:user|>'
      const tokens = collect(input)
      expect(tokens.filter(t => t._tag === 'Open')).toHaveLength(2)
      expect(tokens.filter(t => t._tag === 'Close')).toHaveLength(2)
      expect(tokens.filter(t => t._tag === 'SelfClose')).toHaveLength(1)
    })

    it('accepts the exact LLM example from the user', () => {
      const input = ' <|message:user>\nHey Anders! How can I help you today?\n<message|>\n\n<|yield:user|>'
      const tokens = collect(input)
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'message' && t.variant === 'user')).toBe(true)
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'message')).toBe(true)
      expect(tokens.some(t => t._tag === 'SelfClose' && t.name === 'yield')).toBe(true)
    })
  })
})

// ---------------------------------------------------------------------------
// 4. Grammar DFA leniency — generateBodyRules
// ---------------------------------------------------------------------------

describe('Grammar DFA leniency (generateBodyRules)', () => {
  /**
   * Parse the rules array from generateBodyRules into a map for inspection.
   */
  function parseRules(rules: string[]): Map<string, string> {
    const map = new Map<string, string>()
    for (const rule of rules) {
      const m = rule.match(/^(\S+) ::= (.+)$/)
      if (m) map.set(m[1], m[2])
    }
    return map
  }

  describe('close tag only matches when followed by ws+newline (not mid-line)', () => {
    it('tw0 has escape back to s0 for non-whitespace/non-newline chars', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      const tw0 = map.get('think-body-tw0')
      expect(tw0).toBeDefined()
      // tw0 should allow falling back to s0 when the char after > is not ws/newline
      // e.g. <think|>` should NOT close the block — backtick should escape to s0
      expect(tw0).toContain('think-body-s0')
    })

    it('tw1 has escape back to s0 for non-whitespace/non-newline chars', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      const tw1 = map.get('think-body-tw1')
      expect(tw1).toBeDefined()
      expect(tw1).toContain('think-body-s0')
    })

    it('tw2 has escape back to s0 for non-whitespace/non-newline chars', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      const tw2 = map.get('think-body-tw2')
      expect(tw2).toBeDefined()
      expect(tw2).toContain('think-body-s0')
    })

    it('tw3 has escape back to s0 for non-whitespace/non-newline chars', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      const tw3 = map.get('think-body-tw3')
      expect(tw3).toBeDefined()
      expect(tw3).toContain('think-body-s0')
    })

    it('tw4 has escape back to s0 for non-newline chars (max horizontal ws reached)', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      const tw4 = map.get('think-body-tw4')
      expect(tw4).toBeDefined()
      expect(tw4).toContain('think-body-s0')
    })
  })

  it('generates rules for think body', () => {
    const rules = generateBodyRules('think', 'think')
    expect(rules.length).toBeGreaterThan(0)
    expect(rules[0]).toMatch(/^think-body ::=/)
  })

  it('generates rules for message body', () => {
    const rules = generateBodyRules('msg', 'message')
    expect(rules[0]).toMatch(/^msg-body ::=/)
  })

  it('generates rules for parameter body', () => {
    const rules = generateBodyRules('param', 'parameter')
    expect(rules[0]).toMatch(/^param-body ::=/)
  })

  describe('DFA state structure accepts all 4 close variants', () => {
    it('includes a slash-handling state', () => {
      const rules = generateBodyRules('think', 'think')
      const ruleStr = rules.join('\n')
      // Should have a slash state
      expect(ruleStr).toContain('think-body-sl')
    })

    it('includes a pipe-handling state', () => {
      const rules = generateBodyRules('think', 'think')
      const ruleStr = rules.join('\n')
      expect(ruleStr).toContain('think-body-pp')
    })

    it('s1 state branches on "/" for slash variant', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      const s1 = map.get('think-body-s1')
      expect(s1).toBeDefined()
      expect(s1).toContain('think-body-sl')
    })

    it('slash state leads to s2 (shared tagname tracking)', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      const slash = map.get('think-body-sl')
      expect(slash).toBeDefined()
      // Should reference s2 (first char of tagname)
      expect(slash).toContain('think-body-s2')
    })

    it('final state after full tagname accepts "|" (pipe) and ">" (no-pipe)', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      // tagname "think" has 5 chars, so s1 + 5 = s6 is the final state
      const finalState = map.get('think-body-s6')
      expect(finalState).toBeDefined()
      expect(finalState).toContain('think-body-pp')
      expect(finalState).toContain('">"')
    })

    it('pipe state accepts ">" to terminate', () => {
      const rules = generateBodyRules('think', 'think')
      const map = parseRules(rules)
      const pipeState = map.get('think-body-pp')
      expect(pipeState).toBeDefined()
      expect(pipeState).toContain('">"')
    })

    it('all rules reference s0 as fallback (non-matching chars loop back)', () => {
      const rules = generateBodyRules('think', 'think')
      const ruleStr = rules.join('\n')
      // s0 is the base accumulation state
      expect(ruleStr).toContain('think-body-s0')
    })
  })

  describe('generateBodyRules for "message" (7 chars)', () => {
    it('includes correct number of states', () => {
      const rules = generateBodyRules('msg', 'message')
      const map = parseRules(rules)
      // "message" has 7 chars: s1 (after <) + s2..s8 (7 tagname chars) + s8 final
      // s(n+1) where n=7 → s8 is the final state
      expect(map.has('msg-body-s0')).toBe(true)
      expect(map.has('msg-body-s8')).toBe(true)
      expect(map.has('msg-body-sl')).toBe(true)
      expect(map.has('msg-body-pp')).toBe(true)
    })

    it('final state (s8) accepts pipe and no-pipe termination', () => {
      const rules = generateBodyRules('msg', 'message')
      const map = parseRules(rules)
      // s(n+1) = s(7+1) = s8
      const s8 = map.get('msg-body-s8')
      expect(s8).toBeDefined()
      expect(s8).toContain('msg-body-pp')
      expect(s8).toContain('">"')
    })
  })
})

// ---------------------------------------------------------------------------
// 5. Grammar newline enforcement
// ---------------------------------------------------------------------------

describe('Grammar newline enforcement', () => {
  function buildGrammar(): string {
    return GrammarBuilder.create([
      {
        tagName: 'shell',
        parameters: [{ name: 'cmd', field: 'cmd', type: 'scalar' }],
      },
    ]).build()
  }

  it('lens open tag includes ws before, mandatory \\n after open, then body', () => {
    const grammar = buildGrammar()
    expect(grammar).toContain('ws "<|think:')
    expect(grammar).toContain('>" "\\n" think-body')
  })

  it('lens close tag is handled by DFA with trailing ws + newline', () => {
    const grammar = buildGrammar()
    // DFA handles close tag AND trailing whitespace via tw0/tw1/tw2 states
    expect(grammar).toContain('think-body-tw0')
    // Outer rule ends directly with think-body (no trailing ws in outer rule)
    expect(grammar).toContain('>" think-body')
  })

  it('message open tag includes ws before, mandatory \\n after open, then body', () => {
    const grammar = buildGrammar()
    expect(grammar).toContain('ws "<|message:')
    expect(grammar).toContain('>" "\\n" msg-body')
  })

  it('message close tag is handled by DFA with trailing ws + newline', () => {
    const grammar = buildGrammar()
    // DFA has trailing whitespace states (tw0/tw1/tw2)
    expect(grammar).toContain('msg-body-tw0')
  })

  it('invoke open tag includes ws before, mandatory \\n after open', () => {
    const grammar = buildGrammar()
    expect(grammar).toContain('ws "<|invoke:shell>" "\\n"')
  })

  it('invoke close tag uses lenient invoke-end rule', () => {
    const grammar = buildGrammar()
    expect(grammar).toContain('invoke-end')
    expect(grammar).toContain('"<invoke|>" | "</invoke|>" | "</invoke>" | "<invoke>"')
  })

  it('yield tag includes ws before and no trailing content', () => {
    const grammar = buildGrammar()
    expect(grammar).toMatch(/ws "<\|yield:[^>]+\|>"/)
    // yield has no trailing content — it's terminal
    expect(grammar).not.toMatch(/ws "<\|yield:[^>]+\|>" "\\"\\n\\""/)
  })

  it('parameter close tag is handled by body DFA (no duplicate literal)', () => {
    const grammar = buildGrammar()
    expect(grammar).not.toContain('"<parameter|>" hws')
  })

  it('parameter open tag goes straight into body (inline)', () => {
    const grammar = buildGrammar()
    expect(grammar).toContain('"<|parameter:cmd>" param-body')
    expect(grammar).not.toContain('"\\n" hws "<|parameter:')
  })

  describe('custom toolKeyword', () => {
    it('uses custom keyword in invoke open tag', () => {
      const grammar = GrammarBuilder.create([
        { tagName: 'shell', parameters: [] },
      ])
        .withToolKeyword('tool')
        .build()
      expect(grammar).toContain('ws "<|tool:shell>" "\\n"')
    })
  })
})

// ---------------------------------------------------------------------------
// 6. Negative cases — things that should NOT pass leniency
// ---------------------------------------------------------------------------

describe('Leniency rejection (negative cases)', () => {
  describe('closing tags: invalid forms rejected as content', () => {
    it('rejects </ > (space after slash)', () => {
      expect(collect('</ >')).toEqual([{ _tag: 'Content', text: '</ >' }])
    })

    it('rejects </> (no name)', () => {
      expect(collect('</>')).toEqual([{ _tag: 'Content', text: '</>' }])
    })

    it('rejects </123> (digit-start name)', () => {
      expect(collect('</123>')).toEqual([{ _tag: 'Content', text: '</123>' }])
    })

    it('rejects <//think|> (double slash)', () => {
      const tokens = collect('<//think|>')
      // First / starts close tag, second / is not a valid name char → failAsContent
      const content = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
      expect(content).toContain('<//')
      expect(tokens.some(t => t._tag === 'Close')).toBe(false)
    })

    it('rejects </ think|> (space before name)', () => {
      const tokens = collect('</ think|>')
      const content = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
      expect(content).toContain('</')
      expect(tokens.some(t => t._tag === 'Close')).toBe(false)
    })

    it('rejects </|think> (pipe after slash)', () => {
      const tokens = collect('</|think>')
      // / starts close, | is not a valid name-start → failAsContent
      const content = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
      expect(content).toContain('</')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(false)
    })
  })

  describe('opening tags: non-lenient forms rejected', () => {
    it('rejects <think:strategy> (no pipe prefix) — treated as close tag', () => {
      // <think starts close_name phase, : is not valid → failAsContent
      const tokens = collect('\n<think:strategy>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'think')).toBe(false)
    })

    it('rejects <invoke:shell> (no pipe prefix)', () => {
      const tokens = collect('\n<invoke:shell>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'invoke')).toBe(false)
    })

    it('does NOT rewrite <|unknown> when not in knownToolTags', () => {
      const tokens = collect('<|unknown>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'invoke')).toBe(false)
      expect(tokens).toEqual([{ _tag: 'Open', name: 'unknown', variant: undefined }])
    })

    it('does NOT rewrite <|shell> when knownToolTags is empty', () => {
      const tokens = collect('<|shell>', new Set())
      expect(tokens).toEqual([{ _tag: 'Open', name: 'shell', variant: undefined }])
    })
  })

  describe('newline enforcement: top-level tags without newline rejected', () => {
    it('rejects <|think:strategy> mid-line', () => {
      const tokens = collect('some text <|think:strategy>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'think')).toBe(false)
      const content = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
      expect(content).toContain('<|think:strategy>')
    })

    it('rejects <|message:user> mid-line', () => {
      const tokens = collect('hello <|message:user>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'message')).toBe(false)
    })

    it('rejects <|invoke:shell> mid-line', () => {
      const tokens = collect('run <|invoke:shell>')
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'invoke')).toBe(false)
    })

    it('rejects <|yield:user|> mid-line', () => {
      const tokens = collect('done <|yield:user|>')
      expect(tokens.some(t => t._tag === 'SelfClose' && t.name === 'yield')).toBe(false)
    })

    it('rejects <think|> mid-line (close tag)', () => {
      const tokens = collect('body<think|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(false)
    })

    it('rejects </think|> mid-line (lenient close, no newline)', () => {
      const tokens = collect('body</think|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(false)
    })

    it('rejects </message> mid-line (lenient close Mode 2, no newline)', () => {
      const tokens = collect('body</message>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'message')).toBe(false)
    })

    it('parameter tags ARE accepted mid-line (exempt)', () => {
      expect(collect('text<|parameter:foo>')).toEqual([
        { _tag: 'Content', text: 'text' },
        { _tag: 'Parameter', name: 'foo' },
      ])
      expect(collect('text<parameter|>')).toEqual([
        { _tag: 'Content', text: 'text' },
        { _tag: 'Close', name: 'parameter', pipe: undefined },
      ])
    })
  })

  describe('combined: lenient form + missing newline = double rejection', () => {
    it('rejects </think|> without preceding newline', () => {
      const tokens = collect('content</think|>')
      expect(tokens.some(t => t._tag === 'Close')).toBe(false)
      const content = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
      expect(content).toContain('</think|>')
    })

    it('rejects <message> without preceding newline (Mode 3)', () => {
      const tokens = collect('content<message>')
      expect(tokens.some(t => t._tag === 'Close')).toBe(false)
    })

    it('accepts </think|> WITH preceding newline (lenient + newline = ok)', () => {
      const tokens = collect('content\n</think|>')
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(true)
    })
  })
})
