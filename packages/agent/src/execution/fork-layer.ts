/**
 * ForkLayer type — the services provided by a fork-scoped layer.
 *
 * Lists the identifiers that makeForkLayers + ProjectionReaderTag merge provides.
 * For class-based Context.Tag, the identifier is the class itself.
 * For GenericTag<S>, the identifier is S.
 */

import type { Layer } from 'effect'
import type { Fork } from '@magnitudedev/event-core'
import type { ToolInterceptorTag } from './permission-gate'
import type { WorkingDirectoryTag } from './working-directory'
import type { AgentStateReaderTag } from '../tools/fork'
import type { AgentRegistryStateReaderTag } from '../tools/agent-registry-reader'
import type { WindowStateReaderTag } from '../tools/window-reader'
import type { ConversationStateReaderTag } from '../tools/memory-reader'
import type { TaskGraphStateReaderTag } from '../tools/task-reader'
import type { PolicyContextProviderTag } from '../agents/types'
import type { ProjectionReaderTag } from '../observables/projection-reader'
import type { ChatPersistence } from '../persistence/chat-persistence-service'
import type { ShadowVcs } from '@magnitudedev/vcs'

/**
 * Union of all service identifiers provided by a fork-scoped layer.
 */
export type ForkLayerServices =
  | Fork.ForkContextService
  | AgentRegistryStateReaderTag
  | ConversationStateReaderTag
  | TaskGraphStateReaderTag
  | AgentStateReaderTag
  | WindowStateReaderTag
  | WorkingDirectoryTag
  | PolicyContextProviderTag
  | ToolInterceptorTag
  | ChatPersistence
  | ProjectionReaderTag
  | ShadowVcs

/**
 * The fork-scoped layer type used by execution manager and cortex.
 */
export type ForkLayer = Layer.Layer<ForkLayerServices>
