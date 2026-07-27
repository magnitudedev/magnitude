import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import * as FileSystem from "@effect/platform/FileSystem"
import { Console, Effect, Schema } from "effect"
import * as NodeOs from "node:os"
import * as NodePath from "node:path"

const CacheScope = Schema.Literal(
  "all",
  "onboarding",
  "recommendations",
  "assessments",
  "calibration",
)

type CacheScope = typeof CacheScope.Type

const cacheRoot = NodePath.join(NodeOs.homedir(), ".magnitude", "cache")

const targetsFor = (scope: CacheScope): ReadonlyArray<string> => scope === "all"
  ? [cacheRoot]
  : scope === "onboarding"
    ? [
        NodePath.join(cacheRoot, "local-model-recommendations.json"),
        NodePath.join(cacheRoot, "indexes", "assessments"),
      ]
    : scope === "recommendations"
      ? [NodePath.join(cacheRoot, "local-model-recommendations.json")]
      : scope === "assessments"
        ? [NodePath.join(cacheRoot, "indexes", "assessments")]
        : [NodePath.join(cacheRoot, "indexes", "calibrations")]

const clearDerivedCache = Effect.gen(function* () {
  const scope = yield* Schema.decodeUnknown(CacheScope)(process.argv[2] ?? "all")
  const fs = yield* FileSystem.FileSystem
  yield* Effect.forEach(
    targetsFor(scope),
    (target) => fs.remove(target, { recursive: true, force: true }),
    { discard: true },
  )
  yield* Console.log(`Cleared ${scope} derived cache.`)
})

BunRuntime.runMain(clearDerivedCache.pipe(Effect.provide(BunContext.layer)))
