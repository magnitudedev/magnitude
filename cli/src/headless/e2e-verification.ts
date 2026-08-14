import { createHash } from "node:crypto"
import { Either, Schema } from "effect"
import {
  HeadlessSessionIdSchema,
  type HeadlessSessionId,
} from "@magnitudedev/client-common"

type JsonValue =
  | string
  | number
  | boolean
  | null
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue }

const JsonValueSchema: Schema.Schema<JsonValue> = Schema.suspend(() =>
  Schema.Union(
    Schema.String,
    Schema.Number,
    Schema.Boolean,
    Schema.Null,
    Schema.Array(JsonValueSchema),
    Schema.Record({ key: Schema.String, value: JsonValueSchema }),
  ),
)
const JsonObject = Schema.Record({ key: Schema.String, value: JsonValueSchema })
const ChatMessage = Schema.Struct({
  role: Schema.Literal("system", "user", "assistant"),
  content: Schema.NonEmptyString,
})
const StreamOptions = Schema.Struct({
  include_usage: Schema.Literal(true),
})
const ToolFunction = Schema.Struct({
  name: Schema.NonEmptyString,
  description: Schema.NonEmptyString,
  parameters: JsonObject,
})
const FunctionTool = Schema.Struct({
  type: Schema.Literal("function"),
  function: ToolFunction,
})
const BasicRequest = Schema.Struct({
  method: Schema.Literal("POST"),
  pathname: Schema.Literal("/v1/chat/completions"),
  authorization: Schema.Null,
  body: Schema.Struct({
    stream: Schema.Literal(true),
    model: Schema.Literal("fake-model"),
    messages: Schema.NonEmptyArray(ChatMessage),
  }),
})
const AgentRequest = Schema.Struct({
  method: Schema.Literal("POST"),
  pathname: Schema.Literal("/v1/chat/completions"),
  authorization: Schema.Null,
  body: Schema.Struct({
    stream: Schema.Literal(true),
    stream_options: StreamOptions,
    max_tokens: Schema.Literal(8192),
    model: Schema.Literal("fake-model"),
    messages: Schema.NonEmptyArray(ChatMessage),
    tools: Schema.Array(FunctionTool),
    tool_choice: Schema.Literal("auto"),
  }),
})
const TitleRequest = Schema.Struct({
  method: Schema.Literal("POST"),
  pathname: Schema.Literal("/v1/chat/completions"),
  authorization: Schema.Null,
  body: Schema.Struct({
    stream: Schema.Literal(true),
    stream_options: StreamOptions,
    max_tokens: Schema.Literal(100),
    model: Schema.Literal("fake-model"),
    messages: Schema.NonEmptyArray(ChatMessage),
  }),
})
const DurableSessionMetadata = Schema.Struct({
  sessionId: HeadlessSessionIdSchema,
  created: Schema.String,
  updated: Schema.String,
  chatName: Schema.String,
  workingDirectory: Schema.String,
  visibility: Schema.Union(Schema.Literal("draft"), Schema.Literal("visible")),
  initialVersion: Schema.String,
  lastActiveVersion: Schema.String,
  gitBranch: Schema.NullOr(Schema.String),
  firstUserMessage: Schema.String,
  lastMessage: Schema.String,
  messageCount: Schema.Number,
})

export interface FakeInferenceRequestDescription {
  readonly method: unknown
  readonly pathname: unknown
  readonly authorization: unknown
  readonly body: unknown
}

export type FakeInferenceExpectation =
  | {
      readonly kind: "agent"
      readonly prompt: string
      readonly systemText: string
      readonly toolProtocolDigest?: string
    }
  | {
      readonly kind: "title"
      readonly prompt: string
    }

const strictParseOptions = { onExcessProperty: "error" } as const
const isJsonObject = Schema.is(JsonObject)
const isChatMessage = Schema.is(ChatMessage, strictParseOptions)
const isDurableSessionMetadata = Schema.is(DurableSessionMetadata, strictParseOptions)

const expectedToolNames = [
  "read",
  "write",
  "edit",
  "tree",
  "grep",
  "shell",
  "web_fetch",
  "skill",
  "compact",
  "finish_goal",
] as const

const expectedAgentToolProtocolDigest =
  "4be0a2dfb298eaab591f387b887ed857aa1fa6297598be1c90e512dc6a372718"

const canonicalJson = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`
  if (isJsonObject(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`
  }
  return JSON.stringify(value) ?? '"<non-json>"'
}

export const digestFakeInferenceToolProtocol = (tools: readonly unknown[]): string =>
  createHash("sha256")
    .update(canonicalJson(tools.map((tool) =>
      isJsonObject(tool) && "function" in tool ? tool.function : null
    )))
    .digest("hex")

export function validateDurableSessionMetadata(
  metadata: unknown,
  expected: {
    readonly sessionId: HeadlessSessionId
    readonly workingDirectory: string
    readonly prompt: string
  },
): boolean {
  return isDurableSessionMetadata(metadata)
    && metadata.visibility === "visible"
    && metadata.sessionId === expected.sessionId
    && metadata.workingDirectory === expected.workingDirectory
    && metadata.firstUserMessage === expected.prompt
    && metadata.lastMessage === expected.prompt
    && metadata.messageCount === 1
}

const expectedHeadlessSystemMessage = (systemText: string): string =>
  `${systemText}\n\n# Headless Mode\n\nThis session is running in headless mode. There is no user present to interact with. You are operating autonomously — proceed without waiting for approval or confirmation. Make decisions and take action directly. Do not ask questions or seek clarification; use your best judgment to complete the goal.\n`

const expectedTitleMessage = (prompt: string): string =>
  `Generate a concise title for this conversation based on the user's first message.\nThe title should be 3-7 words in sentence case (capitalize only the first word and proper nouns).\nMaximum 50 characters. Focus on the main task or topic.\nOutput only the title text with no quotes, labels, or formatting.\n\nExamples:\nFix login button on mobile\nAdd OAuth authentication\nDebug failing CI tests\nRefactor API client error handling\n\nBad (too vague): Code changes\nBad (too long): Investigate and fix the issue where the login button does not respond on mobile devices\nBad (wrong case): Fix Login Button On Mobile\n\nUser message: "${prompt}"`

const hasExactWrappedPrompt = (content: string, prompt: string): boolean => {
  const message = `<message from="user">${prompt}</message>`
  if (!content.endsWith(`\n${message}`)) return false
  if (content.split("<message from=\"user\">").length !== 2) return false
  const prefix = content.slice(0, -(message.length + 1))
  return /^<session_context>[\s\S]*<\/session_context>\n--- \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ---$/.test(prefix)
}

const classifySchemaFailure = (
  request: FakeInferenceRequestDescription,
  expectation?: FakeInferenceExpectation,
): readonly string[] => {
  if (request.method !== "POST") return ["expected POST"]
  if (request.pathname !== "/v1/chat/completions") {
    return ["expected /v1/chat/completions"]
  }
  if (request.authorization !== null) return ["expected no authorization header"]
  if (!isJsonObject(request.body)) return ["expected a JSON object body"]
  if (request.body.model !== "fake-model") return ["expected model fake-model"]
  if (request.body.stream !== true) return ["expected streaming inference"]
  if (!Array.isArray(request.body.messages) || request.body.messages.length === 0) {
    return ["expected a non-empty messages array"]
  }
  if (!request.body.messages.every(isChatMessage)) {
    return ["expected structurally valid chat messages"]
  }
  if (!request.body.messages.some((message) => isChatMessage(message) && message.role === "user")) {
    return ["expected a user message"]
  }
  return expectation?.kind === "agent"
    ? ["expected the exact agent request fields"]
    : expectation?.kind === "title"
      ? ["expected the exact title request fields"]
      : ["expected the exact request fields"]
}

export function validateFakeInferenceRequest(
  request: FakeInferenceRequestDescription,
  expectation?: FakeInferenceExpectation,
): readonly string[] {
  if (expectation === undefined) {
    const decoded = Schema.decodeUnknownEither(BasicRequest)(request, strictParseOptions)
    if (Either.isLeft(decoded)) return classifySchemaFailure(request)
    return decoded.right.body.messages.some((message) => message.role === "user")
      ? []
      : ["expected a user message"]
  }

  if (expectation.kind === "agent") {
    const decoded = Schema.decodeUnknownEither(AgentRequest)(request, strictParseOptions)
    if (Either.isLeft(decoded)) return classifySchemaFailure(request, expectation)

    const toolsValid = decoded.right.body.tools.length === expectedToolNames.length
      && decoded.right.body.tools.every((tool, index) => tool.function.name === expectedToolNames[index])
      && digestFakeInferenceToolProtocol(decoded.right.body.tools) === (
        expectation.toolProtocolDigest ?? expectedAgentToolProtocolDigest
      )
    if (!toolsValid) return ["expected the exact agent tool protocol"]
    if (
      decoded.right.body.messages.length !== 2
      || decoded.right.body.messages[0]?.role !== "system"
      || decoded.right.body.messages[0].content !== expectedHeadlessSystemMessage(expectation.systemText)
    ) return ["expected the exact headless system message"]
    if (
      decoded.right.body.messages.length !== 2
      || decoded.right.body.messages[1]?.role !== "user"
      || !hasExactWrappedPrompt(decoded.right.body.messages[1].content, expectation.prompt)
    ) return ["expected the exact wrapped user prompt"]
    return []
  }

  const decoded = Schema.decodeUnknownEither(TitleRequest)(request, strictParseOptions)
  if (Either.isLeft(decoded)) return classifySchemaFailure(request, expectation)
  return decoded.right.body.messages.length === 1
    && decoded.right.body.messages[0]?.role === "user"
    && decoded.right.body.messages[0].content === expectedTitleMessage(expectation.prompt)
    ? []
    : ["expected the exact title request message"]
}

export function validateFinalFakeInferenceState(
  inferenceRequests: number,
  rejectedInferenceRequests: readonly string[],
): readonly string[] {
  return [
    inferenceRequests === 2
      ? null
      : `expected exactly two inference requests, received ${inferenceRequests}`,
    rejectedInferenceRequests.length === 0
      ? null
      : `fake inference endpoint rejected requests: ${rejectedInferenceRequests.join("; ")}`,
  ].filter((error): error is string => error !== null)
}
