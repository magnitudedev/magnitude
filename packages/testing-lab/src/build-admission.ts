import { Effect, Option, Schema, Stream } from "effect"
import { validateReleaseManifest } from "@magnitudedev/release/contracts"
import { ArtifactStore } from "./artifact-store"
import { BuildOutput, readManifest } from "./build-output"
import { type DatabaseSession, decodeRow } from "./database"
import { Evidence, InfrastructureFailure, LeaseId, RunPlan } from "./domain"
import { Fence } from "./lease"
import { WorkId } from "./work-identity"
import { type WorkSpec } from "./execution-plan"
import { isNewerVersion } from "@magnitudedev/release"
import { artifactObjects, artifactReleases, ArtifactInput } from "./inputs"
import { SourceManifest, sha256 } from "./snapshot"
import type { WorkClaim, WorkResult } from "./work-store"

const invalid = (message: string) => new InfrastructureFailure({ operation: "build-admission", message })
/** Called under the live work lock after provider cleanup, before publishing a dependency. */
export const admitBuildOutput = (tx: DatabaseSession, claim: WorkClaim, plan: RunPlan, work: WorkSpec, result: WorkResult) => Effect.gen(function* () {
  if (work.kind !== "build" || Option.isNone(result.output) || plan.request.input.kind !== "source" ||
    result.cases.some(test => test.outcome.status !== "passed")) return yield* invalid("Only a successful source producer can publish packages")
  const store = yield* ArtifactStore
  const output = result.output.value
  const source = yield* readManifest(plan.request.input.digest, SourceManifest)
  if (output.sourceDigest !== plan.request.input.digest || output.sourceCommit !== source.commit ||
    output.artifactHost !== work.target.target.artifactHost || output.backend !== work.backend) return yield* invalid("Build output differs from its source, host or backend assignment")
  const input = yield* readManifest(output.artifactDigest, ArtifactInput)
  for (const release of artifactReleases(input)) {
    yield* validateReleaseManifest(release).pipe(Effect.mapError(error => invalid(error.message)))
    const artifacts = release.artifacts
    if (release.sourceCommit !== source.commit || artifacts.some(item => !Option.contains(item.host, output.artifactHost) || item.filename.includes("/") || item.filename.includes("\\")) ||
      !artifacts.some(item => item.kind === "acn") || !artifacts.some(item => item.kind === "icn-base") ||
      (output.backend !== "cpu" && !artifacts.some(item => item.kind === "icn-backend" && Option.contains(item.backend, output.backend))) ||
      work.consumers.some(id => !artifacts.some(item => item.kind === "desktop" && item.filename.endsWith(`.${plan.targets.find(target => target.target.id === id)!.target.packageFormat}`)))) {
      return yield* invalid("Build output is missing required native packages or changed its source identity")
    }
  }
  if (Option.isSome(input.updateAcceptance)) {
    const pair = input.updateAcceptance.value
    if (pair.sourceDigest !== output.sourceDigest || !isNewerVersion(pair.candidate.version, pair.previous.version)) return yield* invalid("Update acceptance pair differs from admitted source or version order")
  }
  const files = artifactObjects(input)
  const owned = yield* tx.query("SELECT digest,bytes FROM lab_objects WHERE owner=$1 AND digest=ANY($2::text[])",
    [plan.request.owner, [output.artifactDigest, ...files.map(item => item.sha256)]])
  const lengths = new Map((yield* decodeRow(Schema.Array(Schema.Struct({ digest: Schema.String, bytes: Schema.NumberFromString })), owned)).map(item => [item.digest, item.bytes]))
  if (!lengths.has(output.artifactDigest) || files.some(item => lengths.get(item.sha256) !== item.bytes)) return yield* invalid("Build package graph is incomplete or has inconsistent lengths")
  const leases = yield* tx.query("SELECT lease_id,state FROM lab_leases WHERE run_id=$1 AND work_id=$2 AND work_fence=$3", [claim.runId, claim.workId, claim.fence])
  const producers = yield* decodeRow(Schema.Array(Schema.Struct({ lease_id: LeaseId, state: Schema.String })), leases)
  if (producers.length !== 1) return yield* invalid("Build output must identify its actual producer allocation")
  // Preserve real compilation outcomes when cleanup fails, without granting consumer input.
  if (result.cleanupErrors.length || producers[0]!.state !== "Released") return { published: Option.none<BuildOutput>(), evidence: Option.none<typeof Evidence.Type>() }
  const Receipt = Schema.Struct({ workId: WorkId, fence: Fence, leaseId: LeaseId, output: BuildOutput })
  const wire = yield* Schema.encode(Schema.parseJson(Receipt))({ workId: claim.workId, fence: claim.fence, leaseId: producers[0]!.lease_id, output }).pipe(Effect.orDie)
  const bytes = new TextEncoder().encode(wire), digest = sha256(bytes)
  yield* store.put(digest, Stream.make(bytes))
  yield* tx.query("INSERT INTO lab_objects(owner,digest,bytes) VALUES($1,$2,$3) ON CONFLICT DO NOTHING", [plan.request.owner, digest, bytes.byteLength])
  yield* tx.query("INSERT INTO lab_inputs(owner,digest,kind) VALUES($1,$2,'artifacts') ON CONFLICT DO NOTHING", [plan.request.owner, output.artifactDigest])
  return { published: Option.some(output), evidence: Option.some(Evidence.make({ path: "evidence/producer-receipt.json", sha256: digest, bytes: bytes.byteLength })) }
})
