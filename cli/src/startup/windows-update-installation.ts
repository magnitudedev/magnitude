import { BunContext } from "@effect/platform-bun"
import { completeWindowsUpdateHandoff, WindowsUpdateHandoffRequest } from "@magnitudedev/daemon-management/desktop-native"
import { Effect, Schema } from "effect"
import { win32 } from "node:path"

export const runWindowsUpdateHandoff = () => Effect.runPromise(Effect.gen(function* () {
  const request = yield* Effect.tryPromise(async () => {
    let buffer = Buffer.alloc(0)
    let decoded: typeof WindowsUpdateHandoffRequest.Type | undefined
    for await (const chunk of process.stdin) {
      buffer = Buffer.concat([buffer, Buffer.from(chunk)])
      if (decoded || buffer.length > 32_768) throw new Error("Invalid update handoff")
      const newline = buffer.indexOf(10)
      if (newline >= 0) {
        if (newline !== buffer.length - 1) throw new Error("Invalid update handoff")
        decoded = Schema.decodeUnknownSync(Schema.parseJson(WindowsUpdateHandoffRequest))(buffer.subarray(0, newline).toString("utf8"))
        if (process.platform !== "win32" || win32.resolve(process.execPath) !== win32.join(decoded.preparedDirectory, "magnitude-update.exe")) {
          throw new Error("Invalid update helper")
        }
        await new Promise<void>((resolve, reject) => process.stdout.write("ready\n", error => error ? reject(error) : resolve()))
      }
    }
    if (!decoded) throw new Error("Missing update handoff")
    return decoded
  })
  yield* completeWindowsUpdateHandoff(request)
}).pipe(Effect.provide(BunContext.layer), Effect.catchAll(() => Effect.sync(() => { process.exitCode = 1 }))))
