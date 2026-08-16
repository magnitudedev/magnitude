import type { Command } from "@commander-js/extra-typings"
import { Effect } from "effect"
import {
  documentationTopics,
  findDocumentationTopic,
} from "../agent-docs/topics"

type DocumentationCommandResult =
  | { readonly _tag: "Success"; readonly output: string }
  | { readonly _tag: "Failure"; readonly error: string }

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
      Effect.runSync(Effect.sync(() => {
        if (result._tag === "Success") {
          process.stdout.write(result.output)
          return
        }

        process.stderr.write(result.error)
        process.exitCode = 1
      }))
    })
}
