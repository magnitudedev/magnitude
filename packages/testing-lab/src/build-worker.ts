import { Effect, Option, Schema, Stream } from "effect"
import { ArtifactStore } from "./artifact-store"
import { BuildOutput, readManifest } from "./build-output"
import { CaseExecutor, CaseObservation, runCases } from "./case-runner"
import { Evidence, InfrastructureFailure } from "./domain"
import { HostInspector } from "./host-inspector"
import { HostObservation } from "./hardware"
import { sha256, SourceManifest } from "./snapshot"
import { SourceBuilder } from "./source-builder"
import { WorkAssignment, WorkResult } from "./work-store"

/** Build-only guest: no installer, app profile, model acquisition or test consumer state. */
export const runBuildWorker = (assignment: WorkAssignment) => Effect.gen(function* () {
  const work = assignment.work
  if (work.kind !== "build" || assignment.input.kind !== "source") return yield* new InfrastructureFailure({ operation: "build-worker", message: "Producer requires immutable source and a build assignment" })
  const objects = yield* ArtifactStore
  const source = yield* readManifest(assignment.input.digest, SourceManifest)
  const host = yield* (yield* HostInspector).inspect(work.target.target)
  const evidence = <A, I>(name: string, schema: Schema.Schema<A, I>, value: A) => Effect.gen(function* () {
    const wire = yield* Schema.encode(Schema.parseJson(schema))(value)
    const bytes = new TextEncoder().encode(wire), digest = sha256(bytes)
    yield* objects.put(digest, Stream.make(bytes))
    return Evidence.make({ path: `evidence/${name}`, sha256: digest, bytes: bytes.byteLength })
  }).pipe(Effect.mapError(error => new InfrastructureFailure({ operation: "build-worker", message: error.message })))
  const identity = yield* evidence("build-source.json", SourceManifest, source)
  const observed = yield* evidence("build-host.json", HostObservation, host)
  const build = yield* (yield* SourceBuilder).prepare(source, assignment.input.digest, work.target.target, work.backend,
    assignment.plan.targets.some(target => work.consumers.includes(target.target.id) && target.cases.some(test => test.suite === "update")))
  let output: Option.Option<BuildOutput> = Option.none()
  const cases = yield* runCases(work.target.target, work.target.cases).pipe(Effect.provideService(CaseExecutor, {
    execute: test => test.id === "P1" ? build.compile : Effect.gen(function* () {
      const packaged = yield* build.package
      const receipt = BuildOutput.make({ sourceDigest: assignment.input.digest, sourceCommit: source.commit,
        artifactDigest: packaged.digest, artifactHost: work.target.target.artifactHost, backend: work.backend })
      const item = yield* evidence("build-output.json", BuildOutput, receipt)
      output = Option.some(receipt)
      return CaseObservation.make({ detail: "Packaged the compiled native source into an immutable artifact graph", evidence: [...packaged.evidence, item] })
    }),
  }))
  return WorkResult.make({ output, cleanupErrors: [], cases: cases.map(test => ({ ...test,
    evidence: [...new Map([...test.evidence, identity, observed, ...build.evidence().filter(item => test.caseId !== "P1" || item.path !== "evidence/build-package.json")].map(item => [item.sha256, item])).values()],
  })) })
})
