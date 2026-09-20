import { Effect, Schema, Stream } from "effect"
import { ArtifactStore } from "./artifact-store"
import { Evidence, LeaseId, Provider, RunId } from "./domain"
import { type Machine } from "./machines"
import { sha256 } from "./snapshot"

export const WorkerDiagnostic = Schema.Struct({ runId: RunId, leaseId: LeaseId,
  provider: Provider, machine: Schema.NonEmptyString, phase: Schema.Literal("initialization", "execution"), output: Schema.String })

export const redactWorkerOutput = (output: string) => output.replace(/https?:\/\/[^\s"'<>]+/g, value => {
  try {
    const url = new URL(value)
    url.username = ""; url.password = ""; url.search = ""; url.hash = ""
    return url.toString()
  } catch { return "[redacted URL]" }
}).replace(/Bearer\s+[^\s"'<>]+/gi, "Bearer [REDACTED]")
  .replace(/\b((?:[a-z0-9]+_)*(?:sig|token|password|authorization))(["']?)\s*[=:]\s*(?:"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[^\s,;"'<>]+)/gi, "$1$2=[REDACTED]")

export const retainWorkerDiagnostic = (machine: Machine, phase: typeof WorkerDiagnostic.Type["phase"], raw: string, secrets: readonly string[] = []) => Effect.gen(function* () {
  for (const secret of secrets) if (secret) raw = raw.replaceAll(secret, "[REDACTED]")
  const output = redactWorkerOutput(raw).slice(-32 * 1024)
  const name = machine.provider === "local" ? machine.root : machine.provider === "spark" ? machine.host : machine.name
  const json = yield* Schema.encode(Schema.parseJson(WorkerDiagnostic))({ provider: machine.provider, machine: name, phase,
    runId: machine.tags.runId, leaseId: machine.tags.leaseId, output })
  const bytes = new TextEncoder().encode(json), digest = sha256(bytes)
  yield* (yield* ArtifactStore).put(digest, Stream.make(bytes))
  return Evidence.make({ path: `evidence/${phase}-${machine.tags.leaseId}.json`, sha256: digest, bytes: bytes.byteLength })
})
