import { describe, expect, it } from "vitest"
import { mkdtemp, readdir, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { fileURLToPath } from "node:url"

const inferenceModules = async () => (await readdir(fileURLToPath(new URL(".", import.meta.url))))
  .filter((name) => name.endsWith(".ts") && !name.endsWith(".test.ts"))
  .filter((name) => /^(inference|hugging-face|custom-models|models-status|model-|drafter-|output)/.test(name))

const commands = [
  ["hardware"],
  ["models", "list"], ["models", "list", "--all"], ["models", "show", "test:gguf:q4"], ["models", "recommend"],
  ["models", "search", "qwen"], ["models", "show", "owner/repo"], ["models", "pull", "owner/repo:Q4_K_M"],
  ["models", "pull", "test:gguf:q4"], ["models", "cancel", "test:gguf:q4"], ["models", "rm", "test:gguf:q4"],
  ["models", "families"], ["models", "load", "test:gguf:q4"], ["models", "stop"],
  ["models", "drafter", "hf:owner/repo/q.gguf", "auto"],
  ["models", "hide", "hf:owner/repo/q.gguf"], ["models", "unhide", "hf:owner/repo/q.gguf"],
  ["models", "folders", "list"], ["models", "folders", "add", "/models"], ["models", "folders", "remove", "/models"],
  ["models", "folders", "enable", "ollama"], ["models", "folders", "disable", "lm-studio"],
  ["models", "pull", "owner/repo:Q4_K_M", "--drafter", "none"],
  ["models", "show", "https://huggingface.co/owner/repo"],
  ["hf", "login"], ["hf", "status"], ["hf", "logout"],
] as const

describe("non-interactive inference commands", () => {
  it("never read standard input or prompt", async () => {
    for (const name of await inferenceModules()) {
      const source = await Bun.file(new URL(`./${name}`, import.meta.url)).text()
      expect(source, name).not.toMatch(/process\.stdin|Bun\.stdin|node:readline|\bprompt\(|\bconfirm\(/)
    }
  })

  it.each(commands.flatMap((command) => [[...command], [...command, "--json"]]))(
    "%j finishes with standard input left open",
    async (...args) => {
      const root = await mkdtemp(join(tmpdir(), "mag-no-prompt-"))
      const child = Bun.spawn([Bun.which("bun")!, fileURLToPath(new URL("../index.ts", import.meta.url)), ...args], {
        stdin: "pipe",
        stdout: "pipe",
        stderr: "pipe",
        env: { ...process.env, MAGNITUDE_DEV_DATA_DIR: join(root, "absent"), MAGNITUDE_DEV_PORT: "11168" },
      })
      try {
        const timeout = new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), 8_000))
        const outcome = await Promise.race([child.exited, timeout])
        expect(outcome).not.toBe("timeout")
        const message = "No Magnitude service is running. Open the Magnitude desktop app or run `magnitude serve`."
        const stderr = await new Response(child.stderr).text()
        if ((args as readonly string[]).includes("--json")) expect(JSON.parse(stderr)).toEqual({ error: { message, code: "no_service_running" } })
        else expect(stderr).toBe(`${message}\n`)
        expect(await new Response(child.stdout).text()).toBe("")
      } finally {
        child.kill()
        await rm(root, { recursive: true, force: true })
      }
    },
    15_000,
  )
})
