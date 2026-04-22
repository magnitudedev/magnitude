/**
 * validate-chain-grammar.ts
 *
 * Generates the proposed XML chain grammar by hand and validates it against
 * the `gbnf` npm package. Tests accept/reject scenarios from the design doc.
 *
 * Run with: cd packages/xml-act && bun run scripts/validate-chain-grammar.ts
 */

import GBNF from 'gbnf'
import { sanitizeForGbnf } from '../src/grammar/__tests__/helpers'

// ─── Config ──────────────────────────────────────────────────────────────────

const TOOLS = [
  { name: 'shell', params: ['command'] },
  { name: 'read', params: ['path'] },
]

const YIELD_TAGS = ['yield_user', 'yield_invoke', 'yield_parent']

const LENS_NAMES = ['alignment', 'tasks', 'turn']

const MAX_TW = 4 // trailing whitespace states (tw0..tw4)

// ─── Grammar Generation ───────────────────────────────────────────────────────

/**
 * Build a DFA body rule set for a tag with the given close-tag name.
 *
 * The DFA matches any content, rejecting false close tags (ones not followed
 * by bounded whitespace then `\n` or `<`).
 *
 * @param prefix    rule name prefix, e.g. "reason-lens"
 * @param closeTag  the close tag name, e.g. "reason"
 * @param confirmRule  rule to invoke after `\n` confirms the close (e.g. "turn-next-lens")
 * @param confirmNoLtRule  rule to invoke after `<` confirms the close (e.g. "turn-next-lens-no-lt")
 */
function buildBodyDfa(
  prefix: string,
  closeTag: string,
  confirmRule: string,
  confirmNoLtRule: string,
): string[] {
  const lines: string[] = []
  const L = closeTag.length

  // s0: base content state
  lines.push(`${prefix}-s0 ::= [^<] ${prefix}-s0 | "<" ${prefix}-s1`)

  // s1: saw "<", check for "/"
  lines.push(`${prefix}-s1 ::= "/" ${prefix}-sl | "<" ${prefix}-s1 | [^/<] ${prefix}-s0`)

  // sl: saw "</", now match the close tag characters one by one
  const fc = closeTag[0]

  // After matching all chars of the tag name, we need ">" before tw0
  const gtState = `${prefix}-gt`
  lines.push(`${gtState} ::= ">" ${prefix}-tw0 | "<" ${prefix}-s1 | [^<>] ${prefix}-s0`)

  if (L === 1) {
    lines.push(`${prefix}-sl ::= "${fc}" ${gtState} | "<" ${prefix}-s1 | [^<${fc}] ${prefix}-s0`)
  } else {
    lines.push(`${prefix}-sl ::= "${fc}" ${prefix}-s2 | "<" ${prefix}-s1 | [^<${fc}] ${prefix}-s0`)
    for (let i = 2; i <= L; i++) {
      const ch = closeTag[i - 1]
      const stateName = `${prefix}-s${i}`
      const nextState = i === L ? gtState : `${prefix}-s${i + 1}`
      lines.push(`${stateName} ::= "${ch}" ${nextState} | "<" ${prefix}-s1 | [^<${ch}] ${prefix}-s0`)
    }
  }

  // tw0..tw{MAX_TW}: trailing whitespace states
  for (let i = 0; i <= MAX_TW; i++) {
    const stateName = `${prefix}-tw${i}`
    if (i < MAX_TW) {
      lines.push(
        `${stateName} ::= [ \\t] ${prefix}-tw${i + 1} | "\\n" ${confirmRule} | "<" ${confirmNoLtRule} | [^ \\t\\n<] ${prefix}-s0`
      )
    } else {
      // Final tw state: no more [ \t] advancement — excessive whitespace → reject back to content
      lines.push(
        `${stateName} ::= "\\n" ${confirmRule} | "<" ${confirmNoLtRule} | [^ \\n<] ${prefix}-s0`
      )
    }
  }

  return lines
}

function buildGrammar(): string {
  const rules: string[] = []

  // ── ws ──────────────────────────────────────────────────────────────────────
  rules.push(`ws ::= [ \\t\\n]*`)

  // ── Attribute rules ─────────────────────────────────────────────────────────
  const quotedValue = `[^"]*`
  rules.push(`reason-attrs ::= " about=\\"" ${quotedValue} "\\""`)
  rules.push(`reason-attrs-opt ::= reason-attrs | ""`)
  rules.push(`msg-attrs ::= " to=\\"" ${quotedValue} "\\""`)
  rules.push(`invoke-attrs ::= " tool=\\"" ${quotedValue} "\\""`)
  rules.push(`param-attrs ::= " name=\\"" ${quotedValue} "\\""`)

  // ── Yield rules ──────────────────────────────────────────────────────────────
  const yieldWithLt = YIELD_TAGS.map(t => `"<${t}/>"`)
  const yieldNoLt = YIELD_TAGS.map(t => `"${t}/>"`)
  rules.push(`yield ::= ${yieldWithLt.join(' | ')}`)
  rules.push(`yield-no-lt ::= ${yieldNoLt.join(' | ')}`)

  // ── Parameter body DFA ───────────────────────────────────────────────────────
  rules.push(...buildBodyDfa('param-body', 'parameter', 'invoke-next', 'invoke-next-no-lt'))

  // ── Filter body DFA ──────────────────────────────────────────────────────────
  rules.push(...buildBodyDfa('filter-body', 'filter', 'invoke-next', 'invoke-next-no-lt'))

  // ── Invoke-internal continuations ────────────────────────────────────────────
  rules.push(`invoke-next ::= ws invoke-item | ws "</invoke>" turn-next-post`)
  rules.push(
    `invoke-next-no-lt ::= "parameter" param-attrs ">" param-body-s0 | "filter>" filter-body-s0 | "/invoke>" turn-next-post`
  )
  rules.push(
    `invoke-item ::= "<parameter" param-attrs ">" param-body-s0 | "<filter>" filter-body-s0`
  )

  // ── Reason body DFA (lens phase) ─────────────────────────────────────────────
  rules.push(...buildBodyDfa('reason-lens-body', 'reason', 'turn-next-lens', 'turn-next-lens-no-lt'))

  // ── Message body DFA (post-lens phase) ───────────────────────────────────────
  rules.push(...buildBodyDfa('msg-body', 'message', 'turn-next-post', 'turn-next-post-no-lt'))

  // ── Lens-phase continuations ─────────────────────────────────────────────────
  rules.push(`turn-next-lens ::= ws turn-item-lens | ws yield`)
  rules.push(
    `turn-next-lens-no-lt ::= "reason" reason-attrs-opt ">" reason-lens-body-s0` +
    ` | "message" msg-attrs ">" msg-body-s0` +
    ` | "invoke" invoke-attrs ">" invoke-next` +
    ` | yield-no-lt`
  )
  rules.push(
    `turn-item-lens ::=` +
    ` "<reason" reason-attrs-opt ">" reason-lens-body-s0` +
    ` | "<message" msg-attrs ">" msg-body-s0` +
    ` | "<invoke" invoke-attrs ">" invoke-next`
  )

  // ── Post-lens continuations ───────────────────────────────────────────────────
  rules.push(`turn-next-post ::= ws turn-item-post | ws yield`)
  rules.push(
    `turn-next-post-no-lt ::= "message" msg-attrs ">" msg-body-s0` +
    ` | "invoke" invoke-attrs ">" invoke-next` +
    ` | yield-no-lt`
  )
  rules.push(
    `turn-item-post ::=` +
    ` "<message" msg-attrs ">" msg-body-s0` +
    ` | "<invoke" invoke-attrs ">" invoke-next`
  )

  // ── Root ──────────────────────────────────────────────────────────────────────
  rules.push(`root ::= turn-next-lens`)

  return rules.join('\n')
}

// ─── Test Harness ─────────────────────────────────────────────────────────────

let passed = 0
let failed = 0

function isEndState(state: any): boolean {
  const items = [...state] as Array<{ type: string }>
  return items.some(x => x.type === 'end')
}

function runTest(name: string, input: string, expect: 'pass' | 'fail', grammar: string) {
  const sanitized = sanitizeForGbnf(grammar)
  let state = GBNF(sanitized)
  let rejected = false
  let rejectPos = -1
  let rejectChar = ''

  for (let i = 0; i < input.length; i++) {
    try {
      state = state.add(input[i])
    } catch {
      rejected = true
      rejectPos = i
      rejectChar = input[i]
      break
    }
  }

  // A complete parse requires reaching an end state, not just consuming all chars
  // without throwing. If we consumed all chars but aren't at end, the input is incomplete.
  const didPass = !rejected && isEndState(state)
  const ok = (expect === 'pass') === didPass

  if (ok) {
    console.log(`  ✅ ${name}`)
    passed++
  } else {
    console.log(`  ❌ ${name}`)
    if (expect === 'pass' && rejected) {
      const ctx = input.slice(Math.max(0, rejectPos - 30), rejectPos)
      const after = input.slice(rejectPos + 1, rejectPos + 30)
      console.log(`     Expected PASS but was REJECTED at pos ${rejectPos}, char ${JSON.stringify(rejectChar)}`)
      console.log(`     Context: ...${JSON.stringify(ctx)} >>> ${JSON.stringify(rejectChar)} <<< ${JSON.stringify(after)}...`)
    } else if (expect === 'pass' && !rejected && !isEndState(state)) {
      console.log(`     Expected PASS but parse is INCOMPLETE (did not reach end state)`)
      console.log(`     Final state: ${JSON.stringify([...state])}`)
    } else if (expect === 'fail' && didPass) {
      console.log(`     Expected REJECT but was ACCEPTED (reached end state)`)
    }
    failed++
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

const grammar = buildGrammar()
console.log('=== Generated Grammar ===\n')
console.log(grammar)
console.log('\n=== Sanitized Grammar ===\n')
const sanitized = sanitizeForGbnf(grammar)
console.log(sanitized)

// Verify grammar parses
try {
  GBNF(sanitized)
  console.log('\n✅ Grammar parsed successfully by gbnf\n')
} catch (e: any) {
  console.error(`\n❌ Grammar failed to parse: ${e.message}\n`)
  process.exit(1)
}

console.log('=== Accept/Reject Tests ===\n')

// ── ACCEPT cases ──────────────────────────────────────────────────────────────

console.log('--- ACCEPT ---')

runTest(
  'Just yield',
  '<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Reason then yield (newline between)',
  '<reason about="turn">\nsome reasoning\n</reason>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Reason then message then yield (newlines)',
  '<reason about="turn">\nsome reasoning\n</reason>\n<message to="user">\nhello\n</message>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'No newline between tags (reason → message)',
  '<reason about="turn">\nreasoning\n</reason><message to="user">\nhello\n</message><yield_user/>',
  'pass',
  grammar
)

runTest(
  'Spaces between tags',
  '<reason about="turn">\nreasoning\n</reason>  <message to="user">\nhello\n</message>  <yield_user/>',
  'pass',
  grammar
)

runTest(
  'Multiple reasons then message',
  '<reason about="alignment">\nthinking\n</reason>\n<reason about="tasks">\nmore thinking\n</reason>\n<message to="user">\nhello\n</message>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Message then invoke with parameter then yield',
  '<message to="user">\nhello\n</message>\n<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Invoke with filter',
  '<message to="user">\nhello\n</message>\n<invoke tool="shell">\n<parameter name="command">ls</parameter>\n<filter>$.stdout</filter>\n</invoke>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Blank line between tags (ws handles multiple newlines)',
  '<reason about="turn">\nreasoning\n</reason>\n\n<message to="user">\nhello\n</message>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'No newline: message → yield',
  '<message to="user">\nhello\n</message><yield_user/>',
  'pass',
  grammar
)

runTest(
  'No newline: invoke → yield',
  '<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke><yield_user/>',
  'pass',
  grammar
)

runTest(
  'Tab between close and next open tag',
  '<reason about="turn">\nreasoning\n</reason>\t<message to="user">\nhello\n</message>\n<yield_user/>',
  'pass',
  grammar
)

// ── REJECT cases ──────────────────────────────────────────────────────────────

console.log('\n--- REJECT ---')

runTest(
  'Reason after message (ordering violation)',
  '<message to="user">\nhello\n</message>\n<reason about="turn">\nreasoning\n</reason>\n<yield_user/>',
  'fail',
  grammar
)

runTest(
  'False close tag followed by prose (should be treated as content, real close later)',
  '<message to="user">\nhello</message> to end your message</message>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Missing yield at end (incomplete turn)',
  '<message to="user">\nhello\n</message>\n',
  'fail',
  grammar
)

runTest(
  'Unknown tag after close',
  '<message to="user">\nhello\n</message>\n<unknown-tag>content</unknown-tag>\n<yield_user/>',
  'fail',
  grammar
)

runTest(
  'Invoke tags inside reason body are treated as content',
  '<reason about="turn">\n<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n</reason>\n<yield_user/>',
  'pass',
  grammar
)

// ── EDGE CASES ────────────────────────────────────────────────────────────────

console.log('\n--- EDGE CASES ---')

// Whitespace edge cases

runTest(
  'Exactly 4 spaces after close tag then next tag (tw0→tw1→tw2→tw3→tw4, then "<" confirms)',
  '<message to="user">\nhello\n</message>    <invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  // tw4's rule is: "\n" | "<" | [^ \n<] → s0. Space is excluded from [^ \n<],
  // so at tw4 a 5th space has NO valid transition — grammar hard-rejects.
  // The whitespace bound is strictly enforced: 5+ spaces is a grammar error.
  '5 spaces after close tag — grammar hard-rejects at tw4 (no valid transition for 5th space)',
  '<message to="user">\nhello\n</message>     more prose here\n</message>\n<yield_user/>',
  'fail',
  grammar
)

runTest(
  '4 tabs after close tag then newline (tw0→tw1→tw2→tw3→tw4, then "\\n" confirms)',
  '<message to="user">\nhello\n</message>\t\t\t\t\n<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  '5 tabs after close tag (tw4 sees 5th tab → [^ \\n<] matches \\t → s0, close becomes content; real close later)',
  '<message to="user">\nhello\n</message>\t\t\t\t\t<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n</message>\n<yield_user/>',
  'pass',
  grammar
)

// False close tag scenarios

runTest(
  'Close tag mentioned in prose then real close (tw0 sees space then letter → s0; real close confirmed by \\n)',
  '<message to="user">\nThe tag </message> ends a block.\n</message>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Close tag with immediate letter (tw0 sees "f" → [^ \\t\\n<] → s0; real close confirmed by \\n)',
  '<message to="user">\n</message>foo\n</message>\n<yield_user/>',
  'pass',
  grammar
)

// Invoke internal edge cases

runTest(
  'Parameter immediately followed by another parameter (no newline between)',
  '<invoke tool="shell">\n<parameter name="a">val</parameter><parameter name="b">val2</parameter>\n</invoke>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Parameter followed by filter (no newline between)',
  '<invoke tool="shell">\n<parameter name="command">ls</parameter><filter>$.stdout</filter>\n</invoke>\n<yield_user/>',
  'pass',
  grammar
)

runTest(
  'Filter followed by close invoke (no newline between)',
  '<invoke tool="shell">\n<parameter name="command">ls</parameter>\n<filter>$.stdout</filter></invoke>\n<yield_user/>',
  'pass',
  grammar
)

// Yield edge cases

runTest(
  'Yield immediately after close tag (no whitespace at all)',
  '<message to="user">\nhello\n</message><yield_user/>',
  'pass',
  grammar
)

runTest(
  'Yield with 2 spaces before it (tw0→tw1→tw2, then "<" confirms)',
  '<message to="user">\nhello\n</message>  <yield_user/>',
  'pass',
  grammar
)

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n=== Summary: ${passed} passed, ${failed} failed ===`)

// ── Rule count ────────────────────────────────────────────────────────────────

const ruleCount = grammar.split('\n').filter(l => l.includes('::=')).length
console.log(`\n=== Rule count: ${ruleCount} rules ===`)

// ── Scaling analysis ──────────────────────────────────────────────────────────

console.log('\n=== Scaling Analysis ===')
console.log('Rules per body DFA (close tag length L, MAX_TW=4):')
console.log('  - s0: 1')
console.log('  - s1: 1')
console.log('  - sl: 1')
console.log('  - gt (greater-than confirmation): 1')
console.log('  - s2..sL: L-1')
console.log('  - tw0..tw4: 5')
console.log('  Total per DFA: L + 8')
console.log('')

const dfas = [
  { name: 'reason', L: 6 },
  { name: 'message', L: 7 },
  { name: 'invoke', L: 6 },
  { name: 'parameter', L: 9 },
  { name: 'filter', L: 6 },
]
let dfaTotal = 0
for (const { name, L } of dfas) {
  const count = L + 8
  console.log(`  ${name} DFA (L=${L}): ${count} rules`)
  dfaTotal += count
}
console.log(`  Total DFA rules: ${dfaTotal}`)
console.log(`  Fixed rules (ws, attrs, yield, continuations): ~15`)
console.log(`  Grand total: ~${dfaTotal + 15}`)
console.log('')
console.log('Scaling:')
console.log('  More tools: 0 additional rules (tools are free-form quoted values)')
console.log('  More yield tags: 1 alternative per tag in yield/yield-no-lt')
console.log('  maxLenses N: +(L+8) rules per lens slot (one reason-body DFA per slot)')
