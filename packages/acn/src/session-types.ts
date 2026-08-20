import type { CloseableScope } from "effect/Scope"
import type {
  AppEvent,
  CodingAgentSession,
} from "@magnitudedev/agent"
import type {
  DirectoryPath,
  RawMessageUpload,
  RawMentionOccurrence,
  StreamEvent as ProtocolStreamEvent,
} from "@magnitudedev/acn-protocol"

export interface RuntimeEntry {
  readonly id: string
  readonly createdAt: number
  readonly updatedAt: number
  readonly title: string
  readonly cwd: DirectoryPath
  readonly scratchpadPath: string
  readonly session: CodingAgentSession
  readonly scope: CloseableScope
}

export interface SendUserMessageInput {
  readonly sessionId: string
  readonly messageId?: string
  readonly content: string
  readonly taskMode: boolean
  readonly uploads: ReadonlyArray<RawMessageUpload>
  readonly mentions: ReadonlyArray<RawMentionOccurrence>
}

export function hasUserMessageContent(input: Pick<SendUserMessageInput, "content" | "uploads" | "mentions">): boolean {
  return input.content.trim().length > 0 || input.uploads.length > 0 || input.mentions.length > 0
}

export interface SessionExecutionContext {
  readonly cwd: string
  readonly projectRoot: string
  readonly scratchpadPath: string
}

export type UserBashCommandEvent = Extract<AppEvent, { type: "user_bash_command" }>
export type ProtocolDisplayStreamEvent = ProtocolStreamEvent
