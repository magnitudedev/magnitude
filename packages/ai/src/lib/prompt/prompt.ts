import { Schema } from "effect"
import { MessageSchema, type Message } from "./messages"

export interface PromptShape {
  readonly system: readonly string[]
  readonly messages: readonly Message[]
}

export class Prompt extends Schema.Class<Prompt>("Prompt")({
  system: Schema.Array(Schema.String),
  messages: Schema.Array(MessageSchema),
}) {
  static empty(): PromptShape {
    return {
      system: [],
      messages: [],
    }
  }

  static from(messages: readonly Message[], system: readonly string[] = []): PromptShape {
    return {
      system: [...system],
      messages: [...messages],
    }
  }
}
