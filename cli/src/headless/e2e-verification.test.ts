import { describe, expect, it } from "vitest"
import { HeadlessSessionIdSchema } from "@magnitudedev/client-common"
import {
  digestFakeInferenceToolProtocol,
  validateDurableSessionMetadata,
  validateFakeInferenceRequest,
  validateFinalFakeInferenceState,
} from "./e2e-verification"

const valid = {
  method: "POST",
  pathname: "/v1/chat/completions",
  authorization: null,
  body: {
    model: "fake-model",
    stream: true,
    messages: [{ role: "user", content: "Reply with exactly headless-ok." }],
  },
}

describe("headless E2E verification", () => {
  it("accepts only the expected OpenAI-compatible request contract", () => {
    expect(validateFakeInferenceRequest(valid)).toEqual([])
    expect(validateFakeInferenceRequest({ ...valid, method: "GET" })).toContain("expected POST")
    expect(validateFakeInferenceRequest({ ...valid, pathname: "/wrong" })).toContain(
      "expected /v1/chat/completions",
    )
    expect(validateFakeInferenceRequest({ ...valid, body: { ...valid.body, model: "wrong" } })).toContain(
      "expected model fake-model",
    )
    expect(validateFakeInferenceRequest({ ...valid, body: { ...valid.body, stream: false } })).toContain(
      "expected streaming inference",
    )
    expect(validateFakeInferenceRequest({ ...valid, body: { ...valid.body, messages: "forged" } })).toContain(
      "expected a non-empty messages array",
    )
    for (const messages of [
      [42],
      [{}],
      [{ role: "root", content: "forged" }],
      [{ role: "user", content: null }],
    ]) {
      expect(validateFakeInferenceRequest({ ...valid, body: { ...valid.body, messages } })).toContain(
        "expected structurally valid chat messages",
      )
    }
    expect(validateFakeInferenceRequest({
      ...valid,
      body: { ...valid.body, messages: [{ role: "system", content: "system only" }] },
    })).toContain("expected a user message")
    expect(validateFakeInferenceRequest({ ...valid, authorization: "Bearer forged" })).toContain(
      "expected no authorization header",
    )
  })

  it("rejects protocol-valid requests that do not match the exact E2E inference role", () => {
    const prompt = "Reply with exactly headless-ok."
    const systemText = "Do not use tools. Reply with exactly headless-ok."
    const systemMessage = `${systemText}\n\n# Headless Mode\n\nThis session is running in headless mode. There is no user present to interact with. You are operating autonomously — proceed without waiting for approval or confirmation. Make decisions and take action directly. Do not ask questions or seek clarification; use your best judgment to complete the goal.\n`
    const userMessage = `<session_context>\nfixture\n</session_context>\n--- 2026-08-10 12:00:00 ---\n<message from="user">${prompt}</message>`
    const tools = [
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
    ].map((name) => ({
      type: "function",
      function: {
        name,
        description: `${name} fixture`,
        parameters: { type: "object", properties: {}, additionalProperties: false },
      },
    }))
    const agentBody = {
      stream: true,
      stream_options: { include_usage: true },
      max_tokens: 8192,
      model: "fake-model",
      messages: [
        { role: "system", content: systemMessage },
        { role: "user", content: userMessage },
      ],
      tools,
      tool_choice: "auto",
    }
    const request = { ...valid, body: agentBody }
    const expectation = {
      kind: "agent" as const,
      prompt,
      systemText,
      toolProtocolDigest: digestFakeInferenceToolProtocol(tools),
    }

    expect(validateFakeInferenceRequest(request, expectation)).toEqual([])
    expect(validateFakeInferenceRequest({
      ...request,
      body: {
        ...agentBody,
        messages: [
          { role: "system", content: `forged ${systemMessage}` },
          { role: "user", content: userMessage },
        ],
      },
    }, expectation)).toContain("expected the exact headless system message")
    expect(validateFakeInferenceRequest({
      ...request,
      body: {
        ...agentBody,
        messages: [
          { role: "system", content: systemMessage },
          { role: "user", content: userMessage.replace(prompt, `forged ${prompt}`) },
        ],
      },
    }, expectation)).toContain("expected the exact wrapped user prompt")
    expect(validateFakeInferenceRequest({
      ...request,
      body: {
        ...agentBody,
        messages: [...agentBody.messages, { role: "tool", content: prompt }],
      },
    }, expectation)).toContain("expected structurally valid chat messages")
    expect(validateFakeInferenceRequest({
      ...request,
      body: {
        ...agentBody,
        messages: [
          agentBody.messages[0],
          { ...agentBody.messages[1], forged_extra_field: true },
        ],
      },
    }, expectation)).toContain("expected structurally valid chat messages")
    expect(validateFakeInferenceRequest({
      ...request,
      body: { ...agentBody, forged: true },
    }, expectation)).toContain("expected the exact agent request fields")
    expect(validateFakeInferenceRequest({
      ...request,
      body: {
        ...agentBody,
        tools: tools.map((tool, index) => index === 0
          ? {
              ...tool,
              function: {
                ...tool.function,
                parameters: {
                  type: "object",
                  properties: { forged: { type: "string" } },
                  additionalProperties: false,
                },
              },
            }
          : tool),
      },
    }, expectation)).toContain("expected the exact agent tool protocol")
  })

  it("rejects late or rejected inference requests in the final server state", () => {
    expect(validateFinalFakeInferenceState(2, [])).toEqual([])
    expect(validateFinalFakeInferenceState(3, [])).toContain(
      "expected exactly two inference requests, received 3",
    )
    expect(validateFinalFakeInferenceState(2, ["late rejected request"])).toContain(
      "fake inference endpoint rejected requests: late rejected request",
    )
  })

  it("requires exact durable user-message metadata", () => {
    const expected = {
      sessionId: HeadlessSessionIdSchema.make("session-1"),
      workingDirectory: "/repo",
      prompt: "Reply with exactly headless-ok.",
    }
    const metadata = {
      sessionId: expected.sessionId,
      created: "2026-08-13T20:48:56.190Z",
      updated: "2026-08-13T20:49:01.661Z",
      chatName: "headless-ok",
      workingDirectory: expected.workingDirectory,
      visibility: "visible",
      initialVersion: "0.0.2",
      lastActiveVersion: "0.0.2",
      gitBranch: null,
      firstUserMessage: expected.prompt,
      lastMessage: expected.prompt,
      messageCount: 1,
    }

    expect(validateDurableSessionMetadata(metadata, expected)).toBe(true)
    expect(validateDurableSessionMetadata({ ...metadata, visibility: "draft" }, expected)).toBe(false)
    expect(validateDurableSessionMetadata({ ...metadata, messageCount: 2 }, expected)).toBe(false)
    expect(validateDurableSessionMetadata({ ...metadata, lastMessage: "stale" }, expected)).toBe(false)
    expect(validateDurableSessionMetadata({ ...metadata, firstUserMessage: "forged" }, expected)).toBe(false)
    expect(validateDurableSessionMetadata({ ...metadata, forged: true }, expected)).toBe(false)
  })

  it("rejects unknown fields in chat messages", () => {
    expect(validateFakeInferenceRequest(valid)).toEqual([])
    expect(validateFakeInferenceRequest({
      ...valid,
      body: { ...valid.body, messages: [{ role: "user", content: "test", forged_extra: true }] },
    })).toContain("expected structurally valid chat messages")
  })
})
