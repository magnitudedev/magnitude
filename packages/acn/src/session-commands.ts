import { Context, Effect, Layer } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { createId } from "@magnitudedev/generate-id"
import type { AppEvent } from "@magnitudedev/agent"
import { SessionStartFailed, type InterruptTarget, type SessionError } from "@magnitudedev/protocol"
import { AgentRuntime } from "./agent-runtime"
import type {
  SendUserMessageInput,
  SessionExecutionContext,
  UserBashCommandEvent,
} from "./session-types"
import { captureRawImages } from "./attachments/capture-raw-images"
import { collectMentionOccurrences } from "./file-mentions"

export interface SessionCommandsApi {
  readonly sendUserMessage: (input: SendUserMessageInput) => Effect.Effect<void, SessionError>
  readonly sendUserEvent: (
    sessionId: string,
    event: UserBashCommandEvent,
  ) => Effect.Effect<void, SessionError>
  readonly getRuntimeExecutionContext: (
    sessionId: string,
  ) => Effect.Effect<SessionExecutionContext, SessionError>
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

export const SessionCommandsLive: Layer.Layer<
  SessionCommands,
  never,
  AgentRuntime | FileSystem.FileSystem | Path.Path
> = Layer.effect(
  SessionCommands,
  Effect.gen(function* () {
    const runtime = yield* AgentRuntime
    const fs = yield* FileSystem.FileSystem
    const pathService = yield* Path.Path

    const sendUserMessage = Effect.fn("acn.session-commands.send-user-message")(function* (
      input: SendUserMessageInput,
    ) {
      if (
        !input.content.trim() &&
        input.imageAttachments.length === 0 &&
        input.mentions.length === 0
      ) {
        return yield* new SessionStartFailed({
          sessionId: input.sessionId,
          reason: "Message content cannot be empty",
        })
      }

      yield* runtime.withSession(input.sessionId, "send-user-message", (entry) =>
        Effect.gen(function* () {
          const imageParts = yield* captureRawImages({
            scratchpadPath: entry.scratchpadPath,
            attachments: input.imageAttachments,
          }).pipe(
            Effect.provideService(FileSystem.FileSystem, fs),
            Effect.provideService(Path.Path, pathService),
          )
          const mentions = yield* collectMentionOccurrences(
            entry.cwd,
            entry.scratchpadPath,
            input.content,
            input.mentions,
          ).pipe(
            Effect.mapError(
              (error) =>
                new SessionStartFailed({
                  sessionId: input.sessionId,
                  reason: error instanceof Error ? error.message : "Failed to collect mentions",
                }),
            ),
          )
          yield* entry.session.send({
            type: "user_message",
            messageId: input.messageId ?? createId(),
            timestamp: Date.now(),
            forkId: null,
            text: input.content,
            mentions,
            attachments: imageParts.map((image) => ({
              type: "image" as const,
              image,
            })),
            mode: "text",
            synthetic: false,
            taskMode: input.taskMode,
          } satisfies AppEvent)
        }),
      )
    })

    const startGoal = Effect.fn("acn.session-commands.start-goal")(function* (input: {
      readonly sessionId: string
      readonly objective: string
    }) {
      const objective = input.objective.trim()
      if (!objective) {
        return yield* new SessionStartFailed({
          sessionId: input.sessionId,
          reason: "Goal objective cannot be empty",
        })
      }
      yield* runtime.withSession(input.sessionId, "start-goal", (entry) =>
        entry.session.send({
          type: "goal_started",
          forkId: null,
          goalId: createId(),
          objective,
        } satisfies AppEvent),
      )
    })

    return {
      sendUserMessage,
      startGoal,
      getRuntimeExecutionContext: (sessionId) =>
        runtime.withSession(sessionId, "execution-context", (entry) =>
          Effect.succeed({
            cwd: entry.cwd,
            projectRoot: entry.cwd,
            scratchpadPath: entry.scratchpadPath,
          }),
        ),
      sendUserEvent: (sessionId, event) =>
        runtime.withSession(sessionId, "send-user-event", (entry) => entry.session.send(event)),
      interrupt: (sessionId, target) =>
        runtime.withSession(sessionId, "interrupt", (entry) =>
          Effect.gen(function* () {
            if (target._tag === "fork") {
              yield* entry.session.send({
                type: "interrupt",
                forkId: target.forkId,
              } satisfies AppEvent)
              return
            }

            yield* entry.session.send({
              type: "interrupt",
              forkId: null,
            } satisfies AppEvent)
            const agents = yield* entry.session.state.agentStatus.get()
            for (const agent of agents.agents.values()) {
              if (agent.status === "working") {
                yield* entry.session.send({
                  type: "interrupt",
                  forkId: agent.forkId,
                } satisfies AppEvent)
              }
            }
          }),
        ),
    }
  }),
)
