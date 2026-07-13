import { Context, Effect, Layer } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { createId } from "@magnitudedev/generate-id"
import {
  textParts,
  type AppEvent,
  type Attachment as AgentAttachment,
} from "@magnitudedev/agent"
import {
  SessionStartFailed,
  type MessageAttachment,
  type SessionError,
  type InterruptTarget,
} from "@magnitudedev/protocol"
import { AgentRuntime } from "./agent-runtime"
import type { SendUserMessageInput, SessionExecutionContext, UserBashCommandEvent } from "./session-types"
import { materializeRawMessageAttachments } from "./attachments/materialize-raw-message-attachments"

export interface SessionCommandsApi {
  readonly sendUserMessage: (input: SendUserMessageInput) => Effect.Effect<void, SessionError>
  readonly sendUserEvent: (sessionId: string, event: UserBashCommandEvent) => Effect.Effect<void, SessionError>
  readonly getRuntimeExecutionContext: (sessionId: string) => Effect.Effect<SessionExecutionContext, SessionError>
  readonly startGoal: (input: {
    readonly sessionId: string
    readonly objective: string
  }) => Effect.Effect<void, SessionError>
  readonly interrupt: (
    sessionId: string,
    target: InterruptTarget,
  ) => Effect.Effect<void, SessionError>
}

export class SessionCommands extends Context.Tag("SessionCommands")<
  SessionCommands,
  SessionCommandsApi
>() {}

const toAgentAttachment = (attachment: MessageAttachment): AgentAttachment => {
  switch (attachment.type) {
    case "image":
      return attachment
    case "mention_file":
    case "mention_file_range":
    case "mention_directory":
      return attachment
  }
}

export const SessionCommandsLive: Layer.Layer<SessionCommands, never, AgentRuntime | FileSystem.FileSystem | Path.Path> =
  Layer.effect(
    SessionCommands,
    Effect.gen(function* () {
      const runtime = yield* AgentRuntime
      const fs = yield* FileSystem.FileSystem
      const pathService = yield* Path.Path

      const sendUserMessage = Effect.fn("acn.session-commands.send-user-message")(function* (input: SendUserMessageInput) {
        if (!input.content.trim()) {
          return yield* new SessionStartFailed({
            sessionId: input.sessionId,
            reason: "Message content cannot be empty",
          })
        }
        const entry = yield* runtime.requireOrStart(input.sessionId)
        const attachments = yield* materializeRawMessageAttachments({
          cwd: entry.cwd,
          scratchpadPath: entry.scratchpadPath,
          messageContent: input.content,
          attachments: input.attachments,
        }).pipe(
          Effect.provideService(FileSystem.FileSystem, fs),
          Effect.provideService(Path.Path, pathService),
        )
        const event = {
          type: "user_message",
          messageId: input.messageId ?? createId(),
          timestamp: Date.now(),
          forkId: null,
          content: textParts(input.content),
          attachments: attachments.map(toAgentAttachment),
          mode: "text",
          synthetic: false,
          taskMode: input.taskMode,
        } satisfies AppEvent
        yield* entry.session.send(event)
        yield* runtime.touchEntry(input.sessionId)
      })

      const startGoalOnRoot = Effect.fn("acn.session-commands.start-goal-on-root")(function* (entryId: string, objectiveInput: string) {
        const objective = objectiveInput.trim()
        if (!objective) {
          return yield* new SessionStartFailed({
            sessionId: entryId,
            reason: "Goal objective cannot be empty",
          })
        }
        const entry = yield* runtime.requireOrStart(entryId)
        const event = {
          type: "goal_started",
          forkId: null,
          goalId: createId(),
          objective,
        } satisfies AppEvent
        yield* entry.session.send(event)
        yield* runtime.touchEntry(entryId)
      })

      return {
        sendUserMessage,
        getRuntimeExecutionContext: Effect.fn("acn.session-commands.get-runtime-execution-context")(function* (sessionId) {
          const entry = yield* runtime.requireOrStart(sessionId)
          yield* runtime.touchEntry(sessionId)
          return {
            cwd: entry.cwd,
            projectRoot: entry.cwd,
            scratchpadPath: entry.scratchpadPath,
          }
        }),
        sendUserEvent: Effect.fn("acn.session-commands.send-user-event")(function* (sessionId, event) {
          const entry = yield* runtime.requireOrStart(sessionId)
          yield* entry.session.send(event)
          yield* runtime.touchEntry(sessionId)
        }),
        startGoal: Effect.fn("acn.session-commands.start-goal")(function* (input) {
          yield* startGoalOnRoot(input.sessionId, input.objective)
        }),
        interrupt: Effect.fn("acn.session-commands.interrupt")(function* (sessionId, target: InterruptTarget) {
          const entry = yield* runtime.requireOrStart(sessionId)

          if (target._tag === "fork") {
            // Single fork interrupt (null forkId = root)
            const event = { type: "interrupt", forkId: target.forkId } satisfies AppEvent
            yield* entry.session.send(event)
            yield* runtime.touchEntry(sessionId)
            return
          }

          // Interrupt all — fan out to root + every working fork
          const rootInterrupt = { type: "interrupt", forkId: null } satisfies AppEvent
          yield* entry.session.send(rootInterrupt)

          const agentStatus = yield* entry.session.state.agentStatus.get()
          for (const agent of agentStatus.agents.values()) {
            if (agent.status === "working") {
              const event = { type: "interrupt", forkId: agent.forkId } satisfies AppEvent
              yield* entry.session.send(event)
            }
          }
          yield* runtime.touchEntry(sessionId)
        }),
      }
    }),
  )
