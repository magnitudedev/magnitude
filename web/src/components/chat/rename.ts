import { Option } from "effect"
import type { ChatRenamePayload, SessionId } from "@magnitudedev/sdk"

/**
 * A chat's title being edited, with the title it had when editing began: a rename is a
 * compare-and-set from that title, so a rename made meanwhile fails rather than being overwritten.
 */
export interface ChatTitleEdit {
  readonly chatId: SessionId
  readonly from: string
}

/** The rename an edit saves as, unless its title is blank. */
export const renameOf = (edit: ChatTitleEdit, title: string): Option.Option<ChatRenamePayload> =>
  Option.map(Option.liftPredicate(title.trim(), (trimmed) => trimmed !== ""), (trimmed) => ({ ...edit, title: trimmed }))
