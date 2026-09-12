import customEndpointsMarkdown from "./topics/custom-endpoints.md" with { type: "text" }
import cliMarkdown from "./topics/cli.md" with { type: "text" }
import recommendationsMarkdown from "./topics/recommendations.md" with { type: "text" }
import speculativeMethodsMarkdown from "./topics/speculative-methods.md" with { type: "text" }

export interface DocumentationTopic {
  readonly id: string
  readonly description: string
  readonly markdown: string
}

export const documentationTopics: readonly DocumentationTopic[] = [
  {
    id: "cli",
    description: "Use the non-interactive Magnitude CLI",
    markdown: cliMarkdown,
  },
  {
    id: "recommendations",
    description: "Understand local-model recommendation evidence and ranking",
    markdown: recommendationsMarkdown,
  },
  {
    id: "speculative-methods",
    description: "Understand None, MTP, DFlash, and DSpark acceleration",
    markdown: speculativeMethodsMarkdown,
  },
  {
    id: "custom-endpoints",
    description: "Configure an OpenAI-compatible Chat Completions endpoint",
    markdown: customEndpointsMarkdown,
  },
]

export const findDocumentationTopic = (
  id: string,
): DocumentationTopic | undefined =>
  documentationTopics.find((topic) => topic.id === id)
