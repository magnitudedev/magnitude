import { BunContext } from "@effect/platform-bun"
import { completeLinuxUpdateHandoff, installLinuxApplicationUpdate, LinuxUpdateHandoffRequest } from "@magnitudedev/daemon-management/desktop-native"
import { Effect, Schema } from "effect"
import { CLI_VERSION } from "../version"
import { writeSync } from "node:fs"

export const runLinuxUpdateInstallation = (request: string) => Effect.runPromise(
  installLinuxApplicationUpdate(request, CLI_VERSION).pipe(
    Effect.provide(BunContext.layer),
    Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 })),
  ),
)

export const runLinuxUpdateHandoff = () => Effect.runPromise(Effect.gen(function* () {
  // EOF is the retiring owner's lifetime signal, not a PID observation or timeout.
  const request = yield* Effect.tryPromise(async () => {
    let buffer = Buffer.alloc(0)
    let decoded: typeof LinuxUpdateHandoffRequest.Type | undefined
    for await (const chunk of process.stdin) {
      buffer = Buffer.concat([buffer, Buffer.from(chunk)])
      if (decoded || buffer.length > 32_768) throw new Error("Invalid update handoff")
      const newline = buffer.indexOf(10)
      if (newline >= 0) {
        if (newline !== buffer.length - 1) throw new Error("Invalid update handoff")
        decoded = Schema.decodeUnknownSync(Schema.parseJson(LinuxUpdateHandoffRequest))(buffer.subarray(0, newline).toString("utf8"))
        writeSync(3, "ready\n")
      }
    }
    if (!decoded) throw new Error("Missing update handoff")
    return decoded
  })
  yield* completeLinuxUpdateHandoff(request)
}).pipe(Effect.provide(BunContext.layer), Effect.catchAll(() => Effect.sync(() => { process.exitCode = 1 }))))
