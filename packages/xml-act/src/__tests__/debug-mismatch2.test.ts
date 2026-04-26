import { test } from 'vitest'
import { parse, getToolInputs, getStructuralErrors, collectMessageChunks } from './prefix-heuristics/helpers'

test('tool alias mismatch', () => {
  const input = `<magnitude:shell><magnitude:command>pwd</magnitude:command>\n</magnitude:think><magnitude:message>done</magnitude:message><magnitude:yield_user/>`
  const events = parse(input)
  console.log('errors:', JSON.stringify(getStructuralErrors(events), null, 2))
  console.log('tool inputs:', JSON.stringify(getToolInputs(events), null, 2))
  console.log('messages:', collectMessageChunks(events))
})
