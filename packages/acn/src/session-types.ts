import type { CloseableScope } from "effect/Scope"
import type {
  AppEvent,
  CodingAgentSession,
} from "@magnitudedev/agent"
import type {
  RawMessageAttachment,
  StreamEvent as ProtocolStreamEvent,
} from "@magnitudedev/protocol"

export interface RuntimeEntry {
  readonly id: string
  readonly createdAt: number
  readonly updatedAt: number
  readonly title: string
  readonly cwd: string
  readonly scratchpadPath: string
  readonly session: CodingAgentSession
  readonly scope: CloseableScope
}

export interface SendUserMessageInput {
  readonly sessionId: string
  readonly messageId?: string
  readonly content: string
  readonly taskMode: boolean
  readonly attachments: ReadonlyArray<RawMessageAttachment>
}

export interface SessionExecutionContext {
  readonly cwd: string
  readonly projectRoot: string
  readonly scratchpadPath: string
}

export type UserBashCommandEvent = Extract<AppEvent, { type: "user_bash_command" }>
export type ProtocolDisplayStreamEvent = ProtocolStreamEvent
