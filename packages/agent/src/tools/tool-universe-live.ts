import { Layer } from 'effect'
import { ToolUniverseSource } from '../ambient/tool-universe-ambient'
import { toolUniverseToolkit } from './toolkits'

/** Runtime layer supplying the immutable executable tool universe. */
export const ToolUniverseSourceLive = Layer.succeed(ToolUniverseSource, {
  toolkit: toolUniverseToolkit,
})
