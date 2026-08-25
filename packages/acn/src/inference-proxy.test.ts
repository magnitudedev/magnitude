import { describe, expect, it } from "vitest"
import { proxyInferenceWebRequest } from "./server"

const target = {
  origin: new URL("http://127.0.0.1:43210/"),
  clientOptions: { headers: { authorization: "Bearer private-icn" } },
}

describe("ACN inference proxy", () => {
  it("strips the prefix, replaces authority headers, and preserves streaming bodies", async () => {
    const requestBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("request bytes"))
        controller.close()
      },
    })
    const responseBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("response bytes"))
        controller.close()
      },
    })
    let forwardedUrl = ""
    let forwarded: RequestInit | undefined
    const fetchTarget = (async (input: string | URL | Request, init?: RequestInit) => {
      forwardedUrl = String(input)
      forwarded = init
      return new Response(responseBody, {
        status: 206,
        headers: {
          "content-type": "application/octet-stream",
          connection: "keep-alive",
          "x-icn": "preserved",
        },
      })
    }) as typeof fetch
    const signal = new AbortController().signal
    const result = await proxyInferenceWebRequest(new Request(
      "http://127.0.0.1:10100/inference/v1/chat/completions?stream=true",
      {
        method: "POST",
        body: requestBody,
        headers: {
          authorization: "Bearer caller-secret",
          "x-magnitude-acn-id": "rpc-only",
          "content-type": "application/octet-stream",
        },
        // Required by Node's Request implementation; ignored by Bun.
        duplex: "half",
      } as RequestInit,
    ), target, fetchTarget, signal)

    expect(forwardedUrl).toBe("http://127.0.0.1:43210/v1/chat/completions?stream=true")
    expect(new Headers(forwarded?.headers).get("authorization")).toBe("Bearer private-icn")
    expect(new Headers(forwarded?.headers).has("x-magnitude-acn-id")).toBe(false)
    expect(forwarded?.signal).toBe(signal)
    expect(await new Response(forwarded?.body).text()).toBe("request bytes")
    expect(result.status).toBe(206)
    expect(result.headers.get("connection")).toBeNull()
    expect(result.headers.get("x-icn")).toBe("preserved")
    expect(await result.text()).toBe("response bytes")
  })

  it("does not expose the ICN management or metadata surface", async () => {
    let forwarded = false
    const fetchTarget = (async () => {
      forwarded = true
      return new Response("unexpected")
    }) as unknown as typeof fetch
    const result = await proxyInferenceWebRequest(
      new Request("http://127.0.0.1:10100/inference/api/v1/models"),
      target,
      fetchTarget,
    )
    expect(result.status).toBe(404)
    expect(forwarded).toBe(false)
  })
})
