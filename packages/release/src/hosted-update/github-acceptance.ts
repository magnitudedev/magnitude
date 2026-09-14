import { Command, FileSystem, HttpClient, HttpClientRequest } from "@effect/platform"
import { Effect, Schema } from "effect"
import { createHash } from "node:crypto"
import { Stream } from "effect"
import type { UpdateManifest } from "./manifest"
import { HostedCandidateInvalid } from "./release-candidate"

const Release = Schema.Struct({ tag_name: Schema.String, draft: Schema.Boolean, prerelease: Schema.Boolean,
  assets: Schema.Array(Schema.Struct({ name: Schema.String, size: Schema.Number, digest: Schema.NullOr(Schema.String) })) })

/** Test bytes use isolated public prereleases, never production tags or latest. */
export const publishGithubAcceptance = (artifacts: readonly { file: string; manifest: UpdateManifest }[], token: string) => Effect.gen(function* () {
  const first = artifacts[0]?.manifest
  if (!first || first.tag !== `desktop-update-acceptance/${first.commit}/${first.version}` || artifacts.some(a => a.manifest.tag !== first.tag)) {
    return yield* new HostedCandidateInvalid({ message: "Invalid acceptance release namespace" })
  }
  const fs = yield* FileSystem.FileSystem
  for (const { file, manifest } of artifacts) {
    if ((yield* fs.stat(file)).size !== BigInt(manifest.artifact.bytes)) return yield* new HostedCandidateInvalid({ message: "Acceptance size mismatch" })
    const hash = yield* fs.stream(file).pipe(Stream.runFold(createHash("sha256"), (hash, bytes) => hash.update(bytes)))
    if (hash.digest("hex") !== manifest.artifact.sha256) return yield* new HostedCandidateInvalid({ message: "Acceptance digest mismatch" })
  }
  const http = (yield* HttpClient.HttpClient).pipe(HttpClient.withTracerDisabledWhen(() => true))
  const inspect = http.execute(HttpClientRequest.get(`https://api.github.com/repos/magnitudedev/magnitude/releases/tags/${encodeURIComponent(first.tag)}`, {
    headers: { authorization: `Bearer ${token}`, accept: "application/vnd.github+json", "user-agent": "Magnitude-acceptance" },
  }))
  const run = (...args: [string, ...string[]]) => Command.make("gh", ...args).pipe(Command.env({ GH_TOKEN: token }), Command.exitCode)
  const before = yield* inspect
  if (before.status === 404) {
    // Another platform may create the same test cohort concurrently. Inspect authoritative state
    // after either outcome; never overwrite assets when retrying an interrupted preparation.
    yield* run("release", "create", first.tag, "--repo", "magnitudedev/magnitude", "--target", first.commit,
      "--prerelease", "--latest=false", "--title", `Desktop update acceptance ${first.version}`, "--notes", "Isolated updater acceptance artifacts; not a production release")
  } else if (before.status !== 200) return yield* new HostedCandidateInvalid({ message: "Could not inspect acceptance release" })
  const response = yield* inspect
  if (response.status !== 200) return yield* new HostedCandidateInvalid({ message: "Acceptance release was not created" })
  const release = yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(Release)))
  if (release.draft || !release.prerelease || release.tag_name !== first.tag) return yield* new HostedCandidateInvalid({ message: "Wrong acceptance release" })
  for (const { file, manifest } of artifacts) {
    const matches = release.assets.filter(asset => asset.name === manifest.artifact.filename)
    if (matches.length > 0) {
      const asset = matches[0]!
      if (matches.length !== 1 || asset.size !== manifest.artifact.bytes || asset.digest !== `sha256:${manifest.artifact.sha256}`) {
        return yield* new HostedCandidateInvalid({ message: "Existing acceptance asset differs; use a new cohort" })
      }
    } else if ((yield* run("release", "upload", first.tag, file, "--repo", "magnitudedev/magnitude")) !== 0) {
      return yield* new HostedCandidateInvalid({ message: "Acceptance asset upload failed" })
    }
  }
}).pipe(Effect.mapError(error => error instanceof HostedCandidateInvalid ? error : new HostedCandidateInvalid({ message: "Could not publish acceptance artifacts" })))
