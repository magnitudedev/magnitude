import { Schema } from "effect"
import { MessageSchema, type Message, type TerminalMessage } from "./messages"

/**
 * Messages must end with a non-assistant message (UserMessage or ToolResultMessage).
 */
export type TerminalMessages = readonly [...readonly Message[], TerminalMessage]

export class Prompt extends Schema.Class<Prompt>("Prompt")({
  system: Schema.String,
  messages: Schema.Array(MessageSchema),
}) {
  // Override the Schema-inferred type to enforce terminal constraint
  declare readonly messages: TerminalMessages

  static from(args: { system?: string; messages: TerminalMessages }): Prompt {
    return new Prompt({
      system: args.system ?? "",
      messages: [...args.messages],
    })
  }
}
