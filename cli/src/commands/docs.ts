import type { Command } from "@commander-js/extra-typings"
import { Data, Effect, Schema } from "effect"
import {
  documentationTopics,
  findDocumentationTopic,
} from "../agent-docs/topics"
import { runCommand } from "./output"

type DocumentationCommandResult =
  | { readonly _tag: "Success"; readonly output: string }
  | { readonly _tag: "Failure"; readonly error: string }

class DocumentationTopicNotFound extends Data.TaggedError("DocumentationTopicNotFound")<{
  readonly code: string
  readonly message: string
  readonly retryable: boolean
}> {}

const topicLines = (): string =>
  documentationTopics
    .map(({ id, description }) => `  ${id.padEnd(20)} ${description}`)
    .join("\n")

export const renderDocumentationDirectory = (): string => [
  "Magnitude documentation",
  "",
  "Usage: magnitude docs <topic-id>",
  "",
  "Topics:",
  topicLines(),
  "",
].join("\n")

const withOneTrailingNewline = (value: string): string =>
  `${value.replace(/\n+$/, "")}\n`

export const resolveDocumentationCommand = (
  topicId: string | undefined,
): DocumentationCommandResult => {
  if (topicId === undefined) {
    return { _tag: "Success", output: renderDocumentationDirectory() }
  }

  const topic = findDocumentationTopic(topicId)
  if (topic !== undefined) {
    return { _tag: "Success", output: withOneTrailingNewline(topic.markdown) }
  }

  return {
    _tag: "Failure",
    error: [
      `Unknown Magnitude documentation topic: ${topicId}`,
      "",
      "Available topics:",
      ...documentationTopics.map(({ id }) => `  ${id}`),
      "",
    ].join("\n"),
  }
}

export const registerDocsCommand = (program: Command): void => {
  program
    .command("docs")
    .description("Read documentation for Magnitude agents")
    .argument("[topic-id]", "Exact documentation topic ID")
    .allowExcessArguments(false)
    .action((topicId?: string) => {
      const result = resolveDocumentationCommand(topicId)
      const effect = result._tag === "Success"
        ? Effect.succeed(topicId === undefined
            ? { _tag: "Directory" as const, markdown: result.output }
            : { _tag: "Topic" as const, topicId, markdown: result.output })
        : Effect.fail(new DocumentationTopicNotFound({
            code: "documentation_topic_not_found",
            message: result.error.trimEnd(),
            retryable: false,
          }))
      return runCommand({
        effect,
        schema: Schema.Union(
          Schema.TaggedStruct("Directory", { markdown: Schema.String }),
          Schema.TaggedStruct("Topic", {
            topicId: Schema.String,
            markdown: Schema.String,
          }),
        ),
        render: ({ markdown }) => markdown,
      })
    })
}
