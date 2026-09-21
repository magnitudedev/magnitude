import { Schema } from "effect"
import { LabProcessId } from "../application-identity"

export const TerminalInput = Schema.NonEmptyString.pipe(Schema.pattern(/^[^\x00-\x1f\x7f]+$/))
export const TerminalTurn = Schema.Struct({ prompt: TerminalInput, expected: TerminalInput })
export const HarnessTerminalReceipt = Schema.Struct({ sessionId: Schema.NonEmptyString, pid: LabProcessId, model: Schema.NonEmptyString,
  interruptedMessageId: Schema.NonEmptyString, recoveredMessageId: Schema.NonEmptyString, text: Schema.NonEmptyString })

/** These markers observe generation, not letter-case instruction following.
 * Use the same comparison for echo exclusion, display and persisted output. */
export const containsTerminalMarker = (text: string, marker: string): boolean =>
  marker.length > 0 && text.toLowerCase().includes(marker.toLowerCase())

/** TUI renderers can wrap inside a word independently of terminal column width. */
export const screenContainsTerminalMarker = (lines: readonly string[], marker: string): boolean =>
  containsTerminalMarker(lines.map(line => line.trim()).join(""), marker)
