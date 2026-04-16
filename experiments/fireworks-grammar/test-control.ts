const FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
const API_KEY = process.env.FIREWORKS_API_KEY!
const MODEL = "accounts/fireworks/models/glm-5p1"
const MAX_TOKENS = 500

const systemPrompt = await Bun.file("system.txt").text()
console.log(`System prompt: ${systemPrompt.length} chars`)

const grammarFull = await Bun.file("grammar-full.gbnf").text()
console.log(`Grammar: ${grammarFull.length} chars`)

const prompt = "Read the file package.json and tell me about it"

async function runTest(label: string, grammar: string | null) {
  const body: any = {
    model: MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt },
    ],
    max_tokens: MAX_TOKENS,
    reasoning_effort: "none",
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
  const elapsed = Date.now() - start
  const json = await res.json()

  if (!res.ok) {
    console.log(`\n## ${label}`)
    console.log(`ERROR: ${JSON.stringify(json.error)}`)
    return
  }

  const content = json.choices[0].message.content
  const finish = json.choices[0].finish_reason
  const promptTokens = json.usage.prompt_tokens
  const compTokens = json.usage.completion_tokens

  console.log(`\n## ${label}`)
  console.log(`Finish: ${finish} | Tokens: ${promptTokens}/${compTokens} | Time: ${elapsed}ms`)
  console.log(`Has END_TURN: ${content.includes("</END_TURN>")}`)
  console.log(`\nOutput:`)
  console.log(content)
  if (content.length > 600) {
    console.log(`\n... (last 200 chars): ${content.slice(-200)}`)
  }
}

await runTest("CONTROL - No Grammar", null)
await runTest("WITH GRAMMAR - Full Protocol", grammarFull)
