const FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
const API_KEY = process.env.FIREWORKS_API_KEY!
const MODEL = "accounts/fireworks/models/glm-5p1"
const MAX_TOKENS = 500

const systemPrompt = await Bun.file("system.txt").text()
const grammarFull = await Bun.file("grammar.gbnf").text()
console.log(`System prompt: ${systemPrompt.length} chars, Grammar: ${grammarFull.length} chars`)

const prompt = "Read the file package.json and tell me about it"

async function runStream(label: string, grammar: string | null) {
  const body: any = {
    model: MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt },
    ],
    max_tokens: MAX_TOKENS,
    reasoning_effort: "none",
    stream: true,
  }
  if (grammar) {
    body.response_format = { type: "grammar", grammar }
  }

  const start = Date.now()
  const res = await fetch(FIREWORKS_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const json = await res.json()
    console.log(`\n## ${label}`)
    console.log(`ERROR: ${JSON.stringify(json.error)}`)
    return
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let fullContent = ""
  let finishReason = ""
  let chunkCount = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunkCount++
    const text = decoder.decode(value, { stream: true })
    for (const line of text.split("\n")) {
      if (!line.startsWith("data: ")) continue
      const data = line.slice(6).trim()
      if (data === "[DONE]") continue
      try {
        const json = JSON.parse(data)
        const delta = json.choices?.[0]?.delta?.content
        if (delta) fullContent += delta
        if (json.choices?.[0]?.finish_reason) finishReason = json.choices[0].finish_reason
      } catch {}
    }
  }

  const elapsed = Date.now() - start
  console.log(`\n## ${label}`)
  console.log(`Finish: ${finishReason} | Chunks: ${chunkCount} | Time: ${elapsed}ms`)
  console.log(`Has END_TURN: ${fullContent.includes("</END_TURN>")}`)
  console.log(`\nOutput:`)
  console.log(fullContent)
}

//await runStream("CONTROL - No Grammar (streaming)", null)
await runStream("WITH GRAMMAR - Full Protocol (streaming)", grammarFull)