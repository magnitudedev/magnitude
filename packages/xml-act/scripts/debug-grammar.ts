import { GrammarBuilder } from '../src/grammar/grammar-builder'
import { sanitizeForGbnf } from '../src/grammar/__tests__/helpers'
import GBNF from 'gbnf'

const g = GrammarBuilder.create([{tagName:'shell',parameters:[{name:'command',field:'command',type:'scalar'}]}]).build()
const s = sanitizeForGbnf(g)

const input = `<message to="user">\nhello\n</message>\n<reason about="turn">\nthinking\n</reason>\n<yield_user/>`

let state = GBNF(s)
for (let i = 0; i < input.length; i++) {
  state = state.add(input[i])
  if (i >= 25 && i <= 40) {
    const valid = [...state].map((r: any) => r.type + ':' + String.fromCodePoint(...r.value.filter((v: number) => v >= 32 && v < 127)).slice(0, 20))
    console.log(`pos ${i} char=${JSON.stringify(input[i])} valid=[${valid.join(', ')}]`)
  }
}
console.log('ACCEPTED')
