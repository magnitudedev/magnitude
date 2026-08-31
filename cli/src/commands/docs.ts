import type { Command } from "@commander-js/extra-typings"
import { Data, Effect } from "effect"
import {
  documentationTopics,
  findDocumentationTopic,
} from "../agent-docs/topics"
import { renderTable, runCommand } from "./output"

type DocumentationCommandResult =
  | { readonly _tag: "Success"; readonly output: string }
  | { readonly _tag: "Failure"; readonly error: string }

class DocumentationTopicNotFound extends Data.TaggedError("DocumentationTopicNotFound")<{
  readonly message: string
}> {}

export const renderDocumentationDirectory = (): string => [
  "Magnitude documentation",
  "",
  "Usage: magnitude docs <topic-id>",
  "",
  renderTable(documentationTopics, [
    { heading: "TOPIC", value: ({ id }) => id },
    { heading: "DESCRIPTION", value: ({ description }) => description },
  ]).trimEnd(),
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
    .description("Read bundled documentation for Magnitude agents")
    .argument("[topic-id]", "Documentation topic ID from `magnitude docs`")
    .allowExcessArguments(false)
    .action((topicId?: string) => {
      const result = resolveDocumentationCommand(topicId)
      const effect = result._tag === "Success"
        ? Effect.succeed(topicId === undefined
            ? { _tag: "Directory" as const, markdown: result.output }
            : { _tag: "Topic" as const, topicId, markdown: result.output })
        : Effect.fail(new DocumentationTopicNotFound({
            message: result.error.trimEnd(),
          }))
      return runCommand({
        effect,
        render: ({ markdown }) => markdown,
      })
    })
}
