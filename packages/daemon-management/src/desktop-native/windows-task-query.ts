import { randomUUID } from "node:crypto"
import { Duration, Effect, Schema } from "effect"
import { WindowsJobOwner, WindowsPrivatePipes, WindowsPipeName, encodeWindowsCommand } from "@magnitudedev/utils/windows-native"
import { LegacyStartupFailed } from "./legacy-startup-command"
import type { LegacyWindowsStartup } from "./legacy-startup-windows"

export const WindowsLegacyTaskSnapshot = Schema.Union(
  Schema.TaggedStruct("Missing", {}),
  Schema.TaggedStruct("Registered", {
    xml: Schema.NonEmptyString.pipe(Schema.maxLength(65536)),
    currentUserSid: Schema.String.pipe(Schema.pattern(/^S-1-(?:\d+-)+\d+$/), Schema.brand("WindowsUserSid")),
  }),
)
const Reply = Schema.Union(WindowsLegacyTaskSnapshot, Schema.TaggedStruct("Failed", {
  hresult: Schema.Int.pipe(Schema.between(0, 0xffffffff)),
}))

/** Give this query its own application-scoped job owner, separate from the service job. */
export const makeWindowsLegacyTaskControl = (options: {
  readonly executable: string
  readonly environment: Readonly<Record<string, string | undefined>>
  readonly timeout?: Duration.DurationInput
}) => Effect.gen(function* () {
  const jobs = yield* WindowsJobOwner
  const pipes = yield* WindowsPrivatePipes
  const fail = (message: string) => new LegacyStartupFailed({ message })
  const invoke = (arguments_: readonly string[]) => Effect.scoped(Effect.gen(function* () {
    const name = yield* Effect.sync(() => WindowsPipeName.make(`\\\\.\\pipe\\magnitude-task-query-${randomUUID()}`))
    const pipe = yield* pipes.bind(name, true)
    const command = yield* encodeWindowsCommand({ executable: options.executable, arguments: arguments_, environment: options.environment })
    const job = yield* Effect.acquireRelease(jobs.spawn(command, { _tag: "Diagnostics", output: name }),
      job => job.retire("2 seconds").pipe(Effect.catchAll(error => Effect.logError("Task helper cleanup remains unproven; its owner retains the job", error))),
    )
    const output = Effect.gen(function* () {
      yield* pipe.accept
      const chunks: Uint8Array[] = []
      let size = 0
      for (;;) {
        const chunk = yield* pipe.read
        if (chunk.length === 0) return Buffer.concat(chunks)
        size += chunk.length
        if (size > 512 * 1024) return yield* fail("Legacy Windows task query exceeded its output limit")
        chunks.push(chunk)
      }
    })
    const observed = yield* Effect.all([output, job.exit], { concurrency: "unbounded" }).pipe(
      Effect.timeoutFail({ duration: options.timeout ?? "5 seconds", onTimeout: () => fail("Legacy Windows task query timed out") }),
      Effect.exit,
    )
    // A reply or root exit cannot substitute for observed retirement of the helper job.
    yield* job.retire("2 seconds")
    const [bytes, code] = yield* observed
    const source = yield* Effect.try({ try: () => new TextDecoder("utf-8", { fatal: true }).decode(bytes), catch: () => fail("Legacy Windows task query returned invalid UTF-8") })
    const reply = yield* Schema.decodeUnknown(Schema.parseJson(Reply))(source, { onExcessProperty: "error" }).pipe(
      Effect.mapError(() => fail("Legacy Windows task query returned an invalid snapshot")),
    )
    if (reply._tag === "Failed") return yield* fail(`Legacy Windows task operation failed (HRESULT 0x${reply.hresult.toString(16).padStart(8, "0")})`)
    if (code !== 0) return yield* fail(`Legacy Windows task query exited with code ${code}`)
    return reply
  })).pipe(Effect.mapError(error => error._tag === "LegacyStartupFailed" ? error : fail(error.message)))
  return {
    query: invoke([]),
    retire: (registration: LegacyWindowsStartup) => invoke([
      "--retire", registration.digest, registration.principalSid,
    ]).pipe(Effect.flatMap(reply => reply._tag === "Missing" ? Effect.void
      : Effect.fail(fail("Legacy Windows task remains registered after retirement")))),
  }
})
