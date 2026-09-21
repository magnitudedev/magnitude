import { trustWindowsUpdatePublisher } from "./windows-update-publisher"
import { FileSystem } from "@effect/platform"
import { UpdateConfiguration } from "@magnitudedev/release/hosted-update"
import { Effect, Option, Schema, Stream } from "effect"
import { ArtifactStore } from "./artifact-store"
import { AssertionFailure, Target, isWindows } from "./domain"
import { sha256 } from "./snapshot"
import { UpdateAcceptance } from "./update-acceptance"
import { updateFixture, UpdateFixtureAuthority } from "./update-fixture"
import { prepareReleaseUpdatePair } from "./update-pair"

/** Restore only admitted fixture authority; never accept a runtime override of app publisher trust. */
export const prepareUpdateConsumer = (acceptance: UpdateAcceptance, target: Target, directory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const objects = yield* ArtifactStore
  let length = 0
  const chunks = yield* objects.get(acceptance.authority.sha256).pipe(Stream.tap(chunk => Effect.gen(function* () {
    length += chunk.byteLength
    if (length > acceptance.authority.bytes) return yield* new AssertionFailure({ message: "Private update authority exceeds admitted length" })
  })), Stream.runCollect)
  const bytes = Buffer.concat(Array.from(chunks))
  if (bytes.length !== acceptance.authority.bytes || sha256(bytes) !== acceptance.authority.sha256) {
    return yield* new AssertionFailure({ message: "Private update authority differs from admitted bytes" })
  }
  const authority = yield* Schema.decodeUnknown(Schema.parseJson(UpdateFixtureAuthority))(bytes.toString("utf8")).pipe(
    Effect.mapError(() => new AssertionFailure({ message: "Invalid private update authority" })))
  if (isWindows(target.os)) {
    if (Option.isNone(authority.windowsPublisher)) return yield* new AssertionFailure({ message: "Windows update fixture has no admitted signing certificate" })
    yield* trustWindowsUpdatePublisher(authority.windowsPublisher.value)
  }
  yield* fs.makeDirectory(directory, { recursive: true, mode: 0o700 })
  const fixture = yield* updateFixture(directory, authority)
  if (!Schema.equivalence(UpdateConfiguration)(fixture.configuration, acceptance.configuration)) {
    return yield* new AssertionFailure({ message: "Restored update authority does not match package publisher configuration" })
  }
  const pair = yield* prepareReleaseUpdatePair(acceptance.previous, acceptance.candidate, target, directory)
  return { fixture, pair }
})
