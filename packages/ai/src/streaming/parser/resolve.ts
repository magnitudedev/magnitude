import type { JsonFrame, BoundTokenHandler } from './types'
import { bindHandler } from './types'
import { rootHandler } from './handlers/root'
import { objectHandler } from './handlers/object'
import { arrayHandler } from './handlers/array'

export function resolveHandler(frame: JsonFrame): BoundTokenHandler {
  switch (frame.type) {
    case 'root':
      return bindHandler(rootHandler, frame)
    case 'object':
      return bindHandler(objectHandler, frame)
    case 'array':
      return bindHandler(arrayHandler, frame)
  }
}
