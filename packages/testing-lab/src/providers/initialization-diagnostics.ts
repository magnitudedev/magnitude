import { Effect, Schema, Stream } from "effect"
import { ArtifactStore } from "../artifact-store"
import { Evidence, LeaseId, RunId } from "../domain"
import { AzureMachine } from "../machines"
import { sha256 } from "../snapshot"

export const InitializationDiagnostic = Schema.Struct({ runId: RunId, leaseId: LeaseId,
  provider: Schema.Literal("azure"), machine: Schema.NonEmptyString, output: Schema.String })

export const redactInitializationOutput = (output: string) => output.replace(/https?:\/\/[^\s"'<>]+/g, value => {
  try {
    const url = new URL(value)
    url.username = ""; url.password = ""; url.search = ""; url.hash = ""
    return url.toString()
  } catch { return "[redacted URL]" }
}).replace(/Bearer\s+[^\s"'<>]+/gi, "Bearer [REDACTED]")
  .replace(/\b((?:[a-z0-9]+_)*(?:sig|token|password|authorization))(["']?)\s*[=:]\s*(?:"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[^\s,;"'<>]+)/gi, "$1$2=[REDACTED]")

export const retainInitializationDiagnostic = (machine: typeof AzureMachine.Type, raw: string) => Effect.gen(function* () {
  const response = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({
    value: Schema.Array(Schema.Struct({ message: Schema.String })),
  })))(raw)
  const output = redactInitializationOutput(response.value.map(entry => entry.message).join("\n")).slice(-32 * 1024)
  const json = yield* Schema.encode(Schema.parseJson(InitializationDiagnostic))({ provider: "azure", machine: machine.name,
    runId: machine.tags.runId, leaseId: machine.tags.leaseId, output })
  const bytes = new TextEncoder().encode(json), digest = sha256(bytes)
  yield* (yield* ArtifactStore).put(digest, Stream.make(bytes))
  return Evidence.make({ path: `evidence/initialization-${machine.name}.json`, sha256: digest, bytes: bytes.byteLength })
})
