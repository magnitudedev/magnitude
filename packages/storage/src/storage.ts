import * as FileSystem from '@effect/platform/FileSystem'
import * as Path from '@effect/platform/Path'
import { Context, Effect, Layer, Schema } from 'effect'

import { makeAuthStorage } from './auth/storage'
import { makeConfigStorage } from './config/storage'
import { makeLogStorage } from './logs/storage'
import { makeMemoryStorage } from './memory/storage'
import { makeSessionStorage } from './sessions/storage'
import type { AuthStorageShape } from './auth/contracts'
import type { ConfigStorageShape } from './config/contracts'
import type { LogStorageShape } from './logs/contracts'
import type { MemoryStorageShape } from './memory/contracts'
import type { SessionStorageShape } from './sessions/contracts'
import { GlobalStorage } from './services'
import {
  EMPTY_MODEL_STATE,
  EMPTY_ONBOARDING_STATE,
  ModelStateSchema,
  OnboardingStateSchema,
  type ModelState,
  type OnboardingState,
} from './types'
import { makeStateDocument, type StateDocumentError, type StateHandle } from './state'

export interface MagnitudeStorageShape {
  readonly sessions: SessionStorageShape
  readonly auth: AuthStorageShape
  readonly config: ConfigStorageShape
  readonly memory: MemoryStorageShape
  readonly logs: LogStorageShape
  readonly models: StateHandle<ModelState, StateDocumentError>
  readonly onboarding: StateHandle<OnboardingState, StateDocumentError>
}

export class MagnitudeStorage extends Context.Tag('MagnitudeStorage')<
  MagnitudeStorage,
  MagnitudeStorageShape
>() {}

export const StorageLive = Layer.effect(
  MagnitudeStorage,
  Effect.gen(function* () {
    const global = yield* GlobalStorage
    const config = yield* makeConfigStorage()
    const models = yield* makeStateDocument({
      path: global.paths.modelsFile,
      schema: ModelStateSchema,
      initial: () => EMPTY_MODEL_STATE,
      equivalence: Schema.equivalence(ModelStateSchema),
    })
    const onboarding = yield* makeStateDocument({
      path: global.paths.onboardingFile,
      schema: OnboardingStateSchema,
      initial: () => EMPTY_ONBOARDING_STATE,
      equivalence: Schema.equivalence(OnboardingStateSchema),
    })
    return MagnitudeStorage.of({
      sessions: yield* makeSessionStorage(),
      auth: yield* makeAuthStorage(),
      config,
      memory: yield* makeMemoryStorage(),
      logs: yield* makeLogStorage(),
      models,
      onboarding,
    })
  })
)
