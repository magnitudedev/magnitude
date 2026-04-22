import GBNF from 'gbnf'

// Minimal test of recursive alternation for greedy last-match.
// Uses fake tag names (foo/bar) to avoid conflicts with the xml-act parser.

const GRAMMAR = [
  'root ::= "<bar>" body',
  'body ::= buc ("</foo>" buc)* "</foo>"',
  'buc ::= ([^<] | "<" [^/] | "</" [^f] | "</f" [^o] | "</fo" [^o] | "</foo" [^>])*',
].join('\n')

function test(name: string, input: string, shouldPass: boolean) {
  try {
    let state = GBNF(GRAMMAR)
    for (const ch of input) {
      state = state.add(ch)
    }
    if (shouldPass) {
      console.log('PASS: ' + name)
    } else {
      console.log('FAIL: ' + name + ' -- expected rejection but accepted')
    }
  } catch (e: any) {
    if (!shouldPass) {
      console.log('PASS: ' + name + ' (rejected)')
    } else {
      console.log('FAIL: ' + name + ' -- ' + (e.message || '').slice(0, 150))
    }
  }
}

test('basic', '<bar>hello</foo>', true)
test('embedded close', '<bar>has </foo> inside</foo>', true)
test('double embedded', '<bar>a</foo>b</foo>c</foo>', true)
test('empty body', '<bar></foo>', true)

// Token mask inspection: after </foo>, are BOTH paths available?
console.log('\n--- Token mask after first </foo> ---')
let state = GBNF(GRAMMAR)
const prefix = '<bar>hello</foo>'
for (let i = 0; i < prefix.length - 1; i++) {
  state = state.add(prefix[i])
}
// We're now right before the final '>' of </foo>
// Feed the '>' to complete the close tag
state = state.add('>')
// Now we should be at the decision point
const valid = [...state]
console.log('Number of valid token ranges:', valid.length)
for (const rule of valid) {
  if (rule.type === 'CHAR') {
    const chars = (rule.value as number[]).map((v: number) => {
      if (v === 10) return '\\n'
      if (v === 32) return 'SP'
      if (v >= 32 && v < 127) return String.fromCharCode(v)
      return '0x' + v.toString(16)
    })
    console.log('  CHAR: [' + chars.join(', ') + ']')
  } else if (rule.type === 'CHAR_RNG_UPPER') {
    console.log('  RANGE up to: ' + String.fromCharCode(rule.value as number))
  } else {
    console.log('  ' + rule.type + ': ' + JSON.stringify(rule.value))
  }
}
// Key question: can we type a regular content char (like 'X') AND also end-of-input?
console.log('\nTrying to continue with content after </foo>...')
try {
  let s2 = GBNF(GRAMMAR)
  for (const ch of '<bar>hello</foo>X') s2 = s2.add(ch)
  console.log('Content after </foo>: ACCEPTED (both paths live)')
} catch {
  console.log('Content after </foo>: REJECTED (only structural path)')
}

// Verify greedy: the LAST </foo> is structural
console.log('\n--- Greedy last-match verification ---')
try {
  let s3 = GBNF(GRAMMAR)
  // Content has </foo> embedded, then real close
  for (const ch of '<bar>X</foo>Y</foo>') s3 = s3.add(ch)
  // After the second </foo>, can we still continue OR end?
  const v3 = [...s3]
  const hasEnd = v3.some((r: any) => r.type === 'end')
  const hasContent = v3.some((r: any) => r.type === 'char_exclude' || r.type === 'char')
  console.log('After second </foo>: end=' + hasEnd + ' content=' + hasContent)
  console.log('Both paths alive: ' + (hasEnd && hasContent))
} catch (e: any) {
  console.log('ERROR: ' + e.message.slice(0, 100))
}