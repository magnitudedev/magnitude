import type { ImagePart, TextPart } from "./parts"
import type {
  AssistantMessage,
  Message,
  ToolResultMessage,
  UserMessage,
  UserPart,
} from "./messages"
import { Prompt, type TerminalMessages } from "./prompt"

function text(text: string): TextPart {
  return { _tag: "TextPart", text }
}

function toUserParts(input: string | readonly UserPart[]): readonly UserPart[] {
  return typeof input === "string" ? [text(input)] : input
}

export class PromptBuilder<Messages extends readonly Message[] = readonly []> {
  private constructor(
    private readonly systemParts: readonly string[],
    private readonly messages: Messages,
  ) {}

  static empty(): PromptBuilder<readonly []> {
    return new PromptBuilder([], [] as const)
  }

  system(...segments: readonly string[]): PromptBuilder<Messages> {
    return new PromptBuilder([...this.systemParts, ...segments], this.messages)
  }

  user(input: string | readonly UserPart[]): PromptBuilder<readonly [...Messages, UserMessage]> {
    const message: UserMessage = {
      _tag: "UserMessage",
      parts: toUserParts(input),
    }

    return new PromptBuilder(
      this.systemParts,
      [...this.messages, message] as readonly [...Messages, UserMessage],
    )
  }

  assistant(
    message: Omit<AssistantMessage, "_tag">,
  ): PromptBuilder<readonly [...Messages, AssistantMessage]> {
    const next: AssistantMessage = {
      _tag: "AssistantMessage",
      ...message,
    }

    return new PromptBuilder(
      this.systemParts,
      [...this.messages, next] as readonly [...Messages, AssistantMessage],
    )
  }

  toolResult(
    message: Omit<ToolResultMessage, "_tag">,
  ): PromptBuilder<readonly [...Messages, ToolResultMessage]> {
    const next: ToolResultMessage = {
      _tag: "ToolResultMessage",
      ...message,
    }

    return new PromptBuilder(
      this.systemParts,
      [...this.messages, next] as readonly [...Messages, ToolResultMessage],
    )
  }

  build(this: PromptBuilder<readonly [...readonly Message[], UserMessage | ToolResultMessage]>): Prompt {
    return Prompt.from({ system: this.systemParts.join("\n"), messages: this.messages as TerminalMessages })
  }
}
