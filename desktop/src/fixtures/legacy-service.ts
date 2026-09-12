import { Database } from "bun:sqlite"
import { mkdir } from "node:fs/promises"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"

// Disposable legacy fixture. Its separately grouped child intentionally lacks a lifetime guard.
const profile = process.argv[2]!
const engine = Bun.spawn([process.execPath, "-e", "setInterval(()=>{},1000)"], {
  detached: true, stdin: "ignore", stdout: "ignore", stderr: "ignore",
})
const server = Bun.serve({ hostname: "127.0.0.1", port: 0,
  fetch: () => Response.json({ service: "magnitude-acn", pid: process.pid }),
})
const identity = Option.getOrThrow(await Effect.runPromise(ProcessGroupControllerLive.inspect(process.pid)))
await mkdir(join(profile, "acn"), { recursive: true })
const database = new Database(join(profile, "acn/coordination.sqlite"), { create: true })
database.exec("CREATE TABLE owner (id INTEGER, pid INTEGER, process_start_identity TEXT, port INTEGER)")
database.query("INSERT INTO owner VALUES (1, ?, ?, ?)").run(identity.pid, identity.processStartIdentity, server.port!)
database.close()
process.on("SIGTERM", () => process.exit(0))
console.log(JSON.stringify({ pid: process.pid, enginePid: engine.pid }))
