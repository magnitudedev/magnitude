import OpenAI from "openai";
import { readFileSync } from "fs";
import { join } from "path";

const dir = import.meta.dir ?? __dirname;
const system = readFileSync(join(dir, "system.txt"), "utf-8");
const user = readFileSync(join(dir, "user.txt"), "utf-8");
const examples = JSON.parse(readFileSync(join(dir, "examples.json"), "utf-8"));

const client = new OpenAI();

const stream = await client.chat.completions.create({
  model: "gpt-5.4",
  stream: true,
  messages: [
    { role: "system", content: system },
    ...examples,
    { role: "user", content: user },
  ],
});

for await (const chunk of stream) {
  const text = chunk.choices[0]?.delta?.content;
  if (text) process.stdout.write(text);
}

process.stdout.write("\n");
