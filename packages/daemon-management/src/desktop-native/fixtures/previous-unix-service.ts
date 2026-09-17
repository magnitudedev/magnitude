/** Compile as magnitude-service inside an isolated releases/acn tree for hung-service acceptance. */
import { Database } from "bun:sqlite"
import { mkdir, writeFile } from "node:fs/promises"
import { homedir } from "node:os"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"

const home = homedir()
if (!home.includes("magnitude-upgrade-validation")) throw new Error("An isolated test home is required")
process.on("SIGTERM", () => {})
if (process.argv.includes("--engine")) {
  setInterval(() => {}, 1000)
} else {
  const root = join(home, ".magnitude")
  await mkdir(join(root, "acn"), { recursive: true })
  const identity = Option.getOrThrow(await Effect.runPromise(ProcessGroupControllerLive.inspect(process.pid)))
  const db = new Database(join(root, "acn/coordination.sqlite"), { create: true })
  db.exec("CREATE TABLE IF NOT EXISTS owner (id INTEGER PRIMARY KEY, pid INTEGER, process_start_identity TEXT, port INTEGER)")
  db.query("INSERT OR REPLACE INTO owner VALUES (1, ?, ?, 54321)").run(identity.pid, identity.processStartIdentity)
  db.close()
  const engine = Bun.spawn([process.execPath, "--engine"], { detached: true, stdin: "ignore", stdout: "ignore", stderr: "ignore" })
  await writeFile(join(home, "hung-fixture-pids.json"), JSON.stringify([process.pid, engine.pid]))
  Bun.serve({ hostname: "127.0.0.1", port: 10100, fetch: () => Response.json({ service: "magnitude-acn", version: "0.0.15", pid: process.pid, state: { _tag: "Ready" } }) })
}
