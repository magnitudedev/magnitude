const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY
if (!OPENROUTER_API_KEY) { console.error("Set OPENROUTER_API_KEY"); process.exit(1) }

async function test(model: string, max_tokens: number | undefined, label: string) {
  const body: any = {
    model,
    messages: [{ role: "user", content: "Say hi" }],
  }
  if (max_tokens !== undefined) body.max_tokens = max_tokens

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  })
  const data = await res.json()
  if (res.status !== 200) {
    console.log(`${label} → ${res.status}: ${data.error?.message?.slice(0, 120)}`)
  } else {
    console.log(`${label} → OK (provider: ${data.provider}, finish: ${data.choices?.[0]?.finish_reason})`)
  }
}

// Gemma 4 26B - various max_tokens values
await test("google/gemma-4-26b-a4b-it", undefined, "Gemma4 no max_tokens")
await test("google/gemma-4-26b-a4b-it", 262144, "Gemma4 max=262144")
await test("google/gemma-4-26b-a4b-it", 262143, "Gemma4 max=262143")
await test("google/gemma-4-26b-a4b-it", 262000, "Gemma4 max=262000")
await test("google/gemma-4-26b-a4b-it", 131072, "Gemma4 max=131072")
await test("google/gemma-4-26b-a4b-it", 65536, "Gemma4 max=65536")
await test("google/gemma-4-26b-a4b-it", 8192, "Gemma4 max=8192")

// Compare: other models with max_tokens = contextWindow
await test("anthropic/claude-sonnet-4.6", 200000, "Sonnet max=200000")
await test("openai/gpt-4o", 128000, "GPT4o max=128000")

// Gemma 4 streaming
console.log("\n--- Streaming test ---")
const streamRes = await fetch("https://openrouter.ai/api/v1/chat/completions", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "google/gemma-4-26b-a4b-it",
    messages: [{ role: "user", content: "Say hi" }],
    max_tokens: 262144,
    stream: true,
  }),
})
console.log("Stream status:", streamRes.status)
if (streamRes.status !== 200) {
  const errBody = await streamRes.text()
  console.log("Stream error:", errBody.slice(0, 200))
} else {
  const reader = streamRes.body!.getReader()
  const decoder = new TextDecoder()
  let chunks = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks++
    if (chunks <= 3) console.log("Chunk:", decoder.decode(value, { stream: true }).slice(0, 200))
  }
  console.log("Total chunks:", chunks)
}
