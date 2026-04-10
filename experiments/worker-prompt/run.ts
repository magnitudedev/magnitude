import Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "fs";
import { join } from "path";

const dir = import.meta.dir ?? __dirname;
const system = readFileSync(join(dir, "system.txt"), "utf-8");
const user = readFileSync(join(dir, "user.txt"), "utf-8");
const examples = JSON.parse(readFileSync(join(dir, "examples.json"), "utf-8"));

const client = new Anthropic();

const stream = await client.messages.stream({
  model: "claude-sonnet-4-6",
  max_tokens: 8192,
  system,
  messages: [
    ...examples,
    { role: "user", content: user },
  ],
});

for await (const event of stream) {
  if (
    event.type === "content_block_delta" &&
    event.delta.type === "text_delta"
  ) {
    process.stdout.write(event.delta.text);
  }
}

process.stdout.write("\n");
