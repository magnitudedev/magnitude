import { HttpClient, HttpClientRequest } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import type { UpdateManifest } from "./manifest"
import { HostedCandidateInvalid } from "./release-candidate"

const GithubRelease = Schema.Struct({
  tag_name: Schema.String,
  draft: Schema.Boolean,
  assets: Schema.Array(Schema.Struct({ name: Schema.String, size: Schema.Number, state: Schema.String,
    digest: Schema.NullOr(Schema.String) })),
})
const GithubCommit = Schema.Struct({ sha: Schema.String })

/** GitHub's accepted upload is the binary publication; registration reads metadata only. */
export const verifyGithubRelease = (manifests: readonly UpdateManifest[], token: Option.Option<string>) => Effect.gen(function* () {
  const first = manifests[0]
  if (!first || manifests.some(m => m.tag !== first.tag || m.commit !== first.commit || m.version !== first.version)) {
    return yield* new HostedCandidateInvalid({ message: "GitHub registration requires one release" })
  }
  const http = (yield* HttpClient.HttpClient).pipe(HttpClient.withTracerDisabledWhen(() => true))
  const get = (path: string) => http.execute(HttpClientRequest.get(`https://api.github.com/repos/magnitudedev/magnitude/${path}`, {
    headers: { accept: "application/vnd.github+json", "user-agent": "Magnitude-release",
      ...Option.match(token, { onNone: () => ({}), onSome: value => ({ authorization: `Bearer ${value}` }) }) },
  })).pipe(Effect.flatMap(response => Effect.gen(function* () {
    if (response.status !== 200) return yield* new HostedCandidateInvalid({ message: "GitHub release metadata is unavailable" })
    return yield* response.json
  })))
  const release = yield* get(`releases/tags/${encodeURIComponent(first.tag)}`).pipe(Effect.flatMap(Schema.decodeUnknown(GithubRelease)))
  const commit = yield* get(`commits/${encodeURIComponent(first.tag)}`).pipe(Effect.flatMap(Schema.decodeUnknown(GithubCommit)))
  if (release.draft || release.tag_name !== first.tag || commit.sha !== first.commit) {
    return yield* new HostedCandidateInvalid({ message: "GitHub release does not match the accepted public source" })
  }
  for (const manifest of manifests) {
    const matches = release.assets.filter(asset => asset.name === manifest.artifact.filename)
    const asset = matches[0]
    if (matches.length !== 1 || !asset || asset.state !== "uploaded" || asset.size !== manifest.artifact.bytes || asset.digest !== `sha256:${manifest.artifact.sha256}`) {
      return yield* new HostedCandidateInvalid({ message: "GitHub artifact metadata differs from the accepted release" })
    }
  }
}).pipe(Effect.mapError(error => error instanceof HostedCandidateInvalid ? error : new HostedCandidateInvalid({ message: "Could not validate GitHub release metadata" })),
  Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new HostedCandidateInvalid({ message: "GitHub metadata verification timed out" }) }))
