import { Ambient } from '@magnitudedev/event-core'
import { Context, Effect } from 'effect'
import type { Toolkit } from '@magnitudedev/harness'

export interface ToolUniverseSourceService {
  readonly toolkit: Toolkit
}

export class ToolUniverseSource extends Context.Tag('ToolUniverseSource')<
  ToolUniverseSource,
  ToolUniverseSourceService
>() {}

/** Stable executable tools and state models understood by this session runtime. */
export const ToolUniverseAmbient = Ambient.define<Toolkit, ToolUniverseSource>({
  name: 'ToolUniverse',
  initial: Effect.map(ToolUniverseSource, source => source.toolkit),
})
