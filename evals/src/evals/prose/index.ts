/**
 * Prose delimiter eval definition
 */

import type { Eval } from '../../types'
import { ALL_SCENARIOS } from './scenarios'

export const proseEval: Eval = {
  id: 'prose',
  name: 'Prose Delimiter Escaping',
  description: 'Tests whether models correctly use prose delimiters and avoid incorrect escaping within prose-delimited content',
  scenarios: ALL_SCENARIOS
}
