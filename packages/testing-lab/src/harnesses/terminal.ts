import { Schema } from "effect"
import { LabProcessId } from "../application-identity"

export const TerminalInput = Schema.NonEmptyString.pipe(Schema.pattern(/^[^\x00-\x1f\x7f]+$/))
export const TerminalTurn = Schema.Struct({ prompt: TerminalInput, expected: TerminalInput })
export const HarnessTerminalReceipt = Schema.Struct({ sessionId: Schema.NonEmptyString, pid: LabProcessId, model: Schema.NonEmptyString,
  interruptedMessageId: Schema.NonEmptyString, recoveredMessageId: Schema.NonEmptyString, text: Schema.NonEmptyString })
