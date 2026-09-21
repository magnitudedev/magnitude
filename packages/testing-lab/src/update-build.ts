import { createWindowsUpdatePublisher } from "./windows-update-publisher"
import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import { ArtifactStore } from "./artifact-store"
import { snapshotArtifacts } from "./artifact-input"
import { AssertionFailure, Digest, InfrastructureFailure } from "./domain"
import { ArtifactInput } from "./inputs"
import { sha256 } from "./snapshot"
import { UpdateAcceptance } from "./update-acceptance"
import { updateFixture, UpdateFixtureAuthority } from "./update-fixture"

type Invoke = (phase: string, args: readonly string[], environment: Readonly<Record<string, string>>) => Effect.Effect<void, AssertionFailure | InfrastructureFailure>
/** The producer compiles test trust; the clean consumer later restores only its private feed. */
export const buildUpdateAcceptance = (source: Digest, input: typeof ArtifactInput.Type, root: string, objectsDirectory: string, invoke: Invoke) => Effect.scoped(Effect.gen(function* () {
  const objects = yield* ArtifactStore
  const fs = yield* FileSystem.FileSystem
  const fixture = yield* updateFixture(root)
  const publisher = process.platform === "win32" ? Option.some(yield* createWindowsUpdatePublisher) : Option.none()
  const match = /^(\d+)\.(\d+)\.(\d+)/.exec(input.release.version)
  if (!match || !Number.isSafeInteger(Number(match[3]) + 1)) return yield* new AssertionFailure({ message: "Cannot derive updater acceptance versions" })
  const previousVersion = `${match[1]}.${match[2]}.${match[3]}`
  const nextVersion = `${match[1]}.${match[2]}.${Number(match[3]) + 1}`
  const releases = []
  for (const [label, version] of [["previous", previousVersion], ["candidate", nextVersion]] as const) {
    const output = join(root, `update-${label}`)
    yield* invoke(`update-${label}`, ["packages/testing-lab/scripts/build-update-candidate.ts"], {
      ...(Option.isSome(publisher) ? { MAGNITUDE_ACCEPTANCE_WINDOWS_CERTIFICATE: publisher.value.thumbprint, MAGNITUDE_ACCEPTANCE_NSIS: "makensis.exe" } : {}),
      MAGNITUDE_UPDATE_ACCEPTANCE_CONFIG: fixture.configPath, MAGNITUDE_ACCEPTANCE_VERSION: version, MAGNITUDE_ACCEPTANCE_OUTPUT: output,
    })
    const frozen = yield* snapshotArtifacts(join(output, "artifacts/release-manifest.json"), objectsDirectory).pipe(Effect.provideService(FileSystem.FileSystem, fs))
    const built = yield* Schema.decodeUnknown(Schema.parseJson(ArtifactInput))(frozen.json)
    if (built.release.sourceCommit !== input.release.sourceCommit || built.release.version !== version) return yield* new AssertionFailure({ message: "Updater fixture changed its admitted source or version" })
    releases.push(built.release)
  }
  const privateBytes = new TextEncoder().encode(yield* Schema.encode(Schema.parseJson(UpdateFixtureAuthority))({ ...fixture.authority, windowsPublisher: publisher }))
  const digest = sha256(privateBytes)
  yield* objects.put(digest, Stream.make(privateBytes))
  return UpdateAcceptance.make({ sourceDigest: source, previous: releases[0]!, candidate: releases[1]!, configuration: fixture.configuration,
    authority: { sha256: digest, bytes: privateBytes.byteLength } })
}))
