/**
 * Window State Reader Service
 *
 * Tool-scoped access to WindowProjection state without importing projection
 * instances directly into tools.
 */

import { Context, Effect } from 'effect'
import type { ForkWindowState } from '../window'

export interface WindowStateReader {
  readonly getWindowState: (forkId: string | null) => Effect.Effect<ForkWindowState | undefined>
}

export class WindowStateReaderTag extends Context.Tag('WindowStateReader')<
  WindowStateReaderTag,
  WindowStateReader
>() {}
