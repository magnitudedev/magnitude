import GBNF from 'gbnf'
import { sanitizeForGbnf } from '../src/grammar/__tests__/helpers'

const raw = [
  'yield ::= "<yield_user/>"',
  'turn-next-lens ::= ws yield',
  'ws ::= [ \\t\\n]*',
  'reason-lens-body-s0 ::= [^<] reason-lens-body-s0 | "<" reason-lens-body-s1',
  'reason-lens-body-s1 ::= "/" reason-lens-body-sl | "<" reason-lens-body-s1 | [^/<] reason-lens-body-s0',
  'reason-lens-body-sl ::= "r" reason-lens-body-s2 | "<" reason-lens-body-s1 | [^<r] reason-lens-body-s0',
  'reason-lens-body-s2 ::= "e" reason-lens-body-s3 | "<" reason-lens-body-s1 | [^<e] reason-lens-body-s0',
  'reason-lens-body-s3 ::= "a" reason-lens-body-s4 | "<" reason-lens-body-s1 | [^<a] reason-lens-body-s0',
  'reason-lens-body-s4 ::= "s" reason-lens-body-s5 | "<" reason-lens-body-s1 | [^<s] reason-lens-body-s0',
  'reason-lens-body-s5 ::= "o" reason-lens-body-s6 | "<" reason-lens-body-s1 | [^<o] reason-lens-body-s0',
  'reason-lens-body-s6 ::= "n" reason-lens-body-tw0 | "<" reason-lens-body-s1 | [^<n] reason-lens-body-s0',
  'reason-lens-body-tw0 ::= "\\n" turn-next-lens | "<" turn-next-lens | [^ \\t\\n<] reason-lens-body-s0',
  'root ::= reason-lens-body-s0',
].join('\n')

const grammar = sanitizeForGbnf(raw)

try { GBNF(grammar) } catch(e: any) { console.log('grammar error:', e.message.slice(0, 200)); process.exit(1) }

const input = 'hello\n</reason>\n<yield_user/>'
console.log('Input:', JSON.stringify(input))
let state = GBNF(grammar)
for (let i = 0; i < input.length; i++) {
  const ch = input[i]
  try {
    state = state.add(ch)
    const items = [...state]
    const hasEnd = items.some((x: any) => x.type === 'end')
    console.log(`pos ${i} char ${JSON.stringify(ch)}: ${items.length} items, hasEnd=${hasEnd}`)
  } catch(e) {
    console.log('REJECTED at', i, JSON.stringify(ch))
    process.exit(0)
  }
}
console.log('Final:', JSON.stringify([...state]))
