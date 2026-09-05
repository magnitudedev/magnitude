import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect } from "effect"
import { resolve } from "node:path"
import { canonical } from "@magnitudedev/utils/canonical-key"
import {
  awaitPublishedIntegrity,
  publishedPluginIntegrity,
  requireNpm,
  PluginArtifactError,
} from "../packages/release/src/plugin-artifacts"
import { readAcceptedPluginCandidate } from "../packages/release/src/plugin-candidate"
import {
  readPreparedRelease,
  verifyPublicBaseline,
} from "../packages/release/scripts/prepare-release"
import { rpcFingerprint } from "../packages/release/scripts/rpc-fingerprint"
import { publishTag } from "../packages/release/scripts/release-channel"

// Never builds or packs. Only the exact tarballs covered by acceptance can be published.
const program = Effect.gen(function* () {
  const directory = resolve(process.argv[2] ?? "release/integration-candidate")
  const candidate = yield* readAcceptedPluginCandidate(directory)
  const source = yield* readPreparedRelease
  if (
    canonical(source) !== canonical(candidate.plan) ||
    (yield* rpcFingerprint()) !== source.rpc.fingerprint
  ) {
    return yield* new PluginArtifactError({
      message: "Accepted candidate does not match the release source",
    })
  }
  yield* verifyPublicBaseline(source)
  for (const { artifact, publish } of candidate.plan.plugins) {
    const existing = yield* publishedPluginIntegrity(
      artifact.name,
      artifact.version,
      directory
    )
    // A prerelease version publishes under its own dist-tag; `latest` stays stable.
    if (existing === null && publish)
      yield* requireNpm(
        [
          "publish",
          `${directory}/${artifact.filename}`,
          "--access",
          "public",
          "--ignore-scripts",
          "--tag",
          publishTag(artifact.version),
        ],
        directory
      )
    // The registry lags a publish; prove it serves the accepted bytes before reporting success.
    yield* awaitPublishedIntegrity(
      artifact.name,
      artifact.version,
      artifact.integrity,
      directory
    )
    yield* Console.log(`Verified public ${artifact.name}@${artifact.version}`)
  }
})
if (import.meta.main) BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
