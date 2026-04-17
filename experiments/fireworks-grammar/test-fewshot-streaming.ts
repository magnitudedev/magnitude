const FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
const API_KEY = process.env.FIREWORKS_API_KEY!
const MODEL = "accounts/fireworks/models/glm-5p1"
const MAX_TOKENS = 500
const TRIALS = 10

const systemPrompt = await Bun.file("system.txt").text()
const grammarFull = await Bun.file("grammar.gbnf").text()
console.log(`System prompt: ${systemPrompt.length} chars, Grammar: ${grammarFull.length} chars`)

const fewShot = [
  {
    role: "user",
    content: "What's in src/main.ts?",
  },
  {
    role: "assistant",
    content: `<read observe="." path="src/main.ts"/>
<end-turn>
<idle/>
</end-turn>`,
  },
  {
    role: "user",
    content: `<observation tool="read">export function main() { console.log("hi") }</observation>`,
  },
  {
    role: "assistant",
    content: `<message to="user">src/main.ts defines a single \`main\` function that logs "hi" to the console.</message>
<end-turn>
<idle/>
</end-turn>`,
  },
  {
    role: "user",
    content: "Find all TODOs in the repo",
  },
  {
    role: "assistant",
    content: `<grep observe="."><pattern>TODO</pattern></grep>
<end-turn>
<idle/>
</end-turn>`,
  },
  {
    role: "user",
    content: `<observation tool="grep">src/a.ts:12:// TODO: handle errors\nsrc/b.ts:44:// TODO: refactor</observation>`,
  },
  {
    role: "assistant",
    content: `<message to="user">Found 2 TODOs: src/a.ts:12 (handle errors) and src/b.ts:44 (refactor).</message>
<end-turn>
<idle/>
</end-turn>`,
  },
]

const prompt = process.env.PROMPT ?? "Read the file package.json and tell me about it"

async function runStream(label: string, grammar: string | null) {
  const body: any = {
    model: MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      ...fewShot,
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
    return { label, error: JSON.stringify(json.error) }
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
  return { label, finishReason, chunkCount, elapsed, content: fullContent }
}

async function trials() {
  console.log(`\n## ${TRIALS} trials with full grammar + few-shot`)
  const results = await Promise.all(
    Array.from({ length: TRIALS }, (_, i) =>
      runStream(`trial-${i}`, grammarFull),
    ),
  )
  let loops = 0
  for (const [i, r] of results.entries()) {
    if ("error" in r) { console.log(`[${i}] ERROR: ${r.error}`); continue }
    const looped = r.finishReason === "length" || r.content.length > 800
    if (looped) loops++
    console.log(
      `[${i}] finish=${r.finishReason} chunks=${r.chunkCount} time=${r.elapsed}ms ${looped ? "LOOP" : "ok"}`,
    )
    if (looped) console.log(`    ${r.content.replace(/\n/g, "\\n").slice(0, 250)}...`)
  }
  console.log(`\nLOOP RATE: ${loops}/${TRIALS}`)

  for (const [i, r] of results.entries()) {
    if ("content" in r) {
      console.log(`\n--- output [${i}] ---\n${r.content}`)
    }
  }
}

await trials()
