import { Atom } from "@effect-atom/atom-react"
import type {
  ChatConversationId,
  ChatGeneration,
  ChatGenerationId,
  ChatSettings,
} from "@magnitudedev/sdk"

export interface ChatDraft {
  readonly text: string
  readonly settings: Partial<ChatSettings>
  readonly submitted: {
    readonly id: ChatGenerationId
    readonly text: string
  } | null
}
/** Presentation drafts survive navigation without copying the watched conversation. */
export const selectedChat = Atom.keepAlive(
  Atom.make<ChatConversationId | null>(null)
)
export const chatDraft = Atom.family((id: ChatConversationId | null) =>
  Atom.keepAlive(
    Atom.make<ChatDraft>({ text: "", settings: {}, submitted: null })
  )
)

export const completeChatSubmission = (
  draft: ChatDraft,
  text: string,
  id: ChatGenerationId,
  retry: boolean
): ChatDraft => ({
  ...draft,
  text: !retry && draft.text === text ? "" : draft.text,
  submitted: { id, text },
})
/** A failed request restores its captured input only while the next draft is still empty. */
export const chatComposerText = (
  draft: ChatDraft,
  last: ChatGeneration | undefined
): string =>
  draft.text ||
  (last?.status === "failed" && last.id === draft.submitted?.id
    ? draft.submitted.text
    : "")
