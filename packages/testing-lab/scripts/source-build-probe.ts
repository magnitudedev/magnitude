import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Layer, Schema } from "effect"
import { join, resolve } from "node:path"
import { fileArtifactStore } from "../src/artifact-store"
import { CaseExecutor, CaseObservation, runCases } from "../src/case-runner"
import { cases, findTarget } from "../src/catalog"
import { CaseResult, TargetId } from "../src/domain"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"
import { snapshotSource } from "../src/snapshot"
import { nativeSourceBuilder, SourceBuilder } from "../src/source-builder"

BunRuntime.runMain(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = resolve(yield* Config.string("LAB_BUILD_PROBE_ROOT"))
  const target = yield* Config.string("LAB_BUILD_PROBE_TARGET").pipe(Effect.flatMap(Schema.decodeUnknown(TargetId)), Effect.flatMap(findTarget))
  const objects = join(root, "objects")
  const source = yield* snapshotSource(resolve(import.meta.dir, "../../.."), objects)
  const environment = Object.fromEntries(["PATH", "TMPDIR", "USER", "LOGNAME", "SystemRoot", "TEMP"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const results = yield* Effect.gen(function* () {
    const stages = yield* (yield* SourceBuilder).prepare(source.manifest, source.digest, target)
    const results = yield* runCases(target, cases.filter(test => test.id === "P1" || test.id === "P2")).pipe(Effect.provideService(CaseExecutor, {
      execute: test => test.id === "P1" ? stages.compile : stages.package.pipe(Effect.map(result => CaseObservation.make({ detail: "Built and admitted final installer bytes", evidence: result.evidence }))),
    }))
    return results.map(result => ({ ...result, evidence: stages.evidence().filter(item => result.caseId !== "P1" || item.path !== "evidence/build-package.json") }))
  }).pipe(Effect.provide(nativeSourceBuilder({ root: join(root, "build"), objects, environment }).pipe(Layer.provide(fileArtifactStore(objects)))))
  yield* fs.writeFileString(join(root, "result.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ sourceDigest: Schema.String, cases: Schema.Array(CaseResult) })))({ sourceDigest: source.digest, cases: results }))
  if (results.some(result => result.outcome.status !== "passed")) process.exitCode = 1
}).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
