import type { Op } from '../machine'
import type { XmlActEvent, XmlActFrame } from './types'

export type Fx = Op<XmlActFrame, XmlActEvent>

export function emit(event: XmlActEvent): Fx {
  return { type: 'emit', event }
}

export function push(frame: XmlActFrame): Fx {
  return { type: 'push', frame }
}

export const pop: Fx = { type: 'pop' }

export function replace(frame: XmlActFrame): Fx {
  return { type: 'replace', frame }
}

export const done: Fx = { type: 'done' }