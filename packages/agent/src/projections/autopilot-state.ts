import { Projection, Signal } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { AppEvent } from '../events'
import { UserMessageResolutionProjection } from './user-message-resolution'
import { InitialTaskAmbient } from '../ambient/initial-task-ambient'

export const AutopilotStateSchema = Schema.Struct({
  enabled: Schema.Boolean,
  pendingContent: Schema.NullOr(Schema.String),
  generating: Schema.Boolean,
})
export type AutopilotState = typeof AutopilotStateSchema.Type

export const AutopilotStateProjection = Projection.define<AppEvent>()({
  name: 'AutopilotState',
  state: AutopilotStateSchema,

  reads: [UserMessageResolutionProjection] as const,

  ambients: [InitialTaskAmbient] as const,

  initial: {
    enabled: false,
    pendingContent: null,
    generating: false,
  },

  signals: {
    autopilotStateChanged: Signal.create<{
      enabled: boolean
      pendingContent: string | null
      generating: boolean
    }>('AutopilotState/changed'),
  },

  eventHandlers: {
    autopilot_toggled: ({ event, state, emit }) => {
      // TEMPORARILY DISABLED: autopilot toggles are ignored.
      // const next = { ...state, enabled: event.enabled, generating: false }
      const next = { enabled: false, pendingContent: null, generating: false }
      emit.autopilotStateChanged({
        enabled: next.enabled,
        pendingContent: next.pendingContent,
        generating: next.generating,
      })
      return next
    },

    autopilot_generation_started: ({ event, state, emit }) => {
      // TEMPORARILY DISABLED: autopilot generation cannot start.
      // const next = { ...state, generating: true }
      const next = { enabled: false, pendingContent: null, generating: false }
      emit.autopilotStateChanged({
        enabled: next.enabled,
        pendingContent: next.pendingContent,
        generating: next.generating,
      })
      return next
    },

    autopilot_outcome: ({ event, state, emit }) => {
      // TEMPORARILY DISABLED: autopilot output should not create pending content.
      // if (event.result._tag === 'success') {
      //   const next = { ...state, pendingContent: event.result.content, generating: false }
      //   emit.autopilotStateChanged({
      //     enabled: next.enabled,
      //     pendingContent: next.pendingContent,
      //     generating: next.generating,
      //   })
      //   return next
      // }
      // // error
      // const next = { ...state, generating: false }
      const next = { enabled: false, pendingContent: null, generating: false }
      emit.autopilotStateChanged({
        enabled: next.enabled,
        pendingContent: next.pendingContent,
        generating: next.generating,
      })
      return next
    },
  },

  signalHandlers: (on) => [
    on(UserMessageResolutionProjection.signals.userMessageResolved, ({ value, state, emit }) => {
      // When a user message is resolved (same timing as TurnProjection's trigger),
      // the context has changed — clear the preview.
      if (state.pendingContent !== null) {
        const next = { ...state, pendingContent: null, generating: false }
        emit.autopilotStateChanged({
          enabled: next.enabled,
          pendingContent: next.pendingContent,
          generating: next.generating,
        })
        return next
      }
      return state
    }),
  ],
})
