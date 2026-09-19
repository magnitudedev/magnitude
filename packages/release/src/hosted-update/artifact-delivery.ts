import { FetchHttpClient, HttpClient, HttpClientError, HttpClientRequest } from "@effect/platform"
import { Effect, Schema } from "effect"
import { isGithubReleaseAssetUrl } from "./github-artifact"
import { githubDownloadClient } from "./github-download"

const AcceptanceOrigin = Schema.String.pipe(Schema.filter(value => {
  try {
    const url = new URL(value)
    return url.origin === value && url.protocol === "https:" && !url.username && !url.password
      && url.hostname !== "magnitude.dev" && !url.hostname.endsWith(".magnitude.dev")
      && url.hostname !== "github.com" && !url.hostname.endsWith(".githubusercontent.com")
  } catch { return false }
}), Schema.brand("AcceptanceArtifactOrigin"))

/** Private delivery is selected by trusted acceptance-build composition, never by an update response. */
export const ArtifactDelivery = Schema.Union(
  Schema.TaggedStruct("Github", {}),
  Schema.TaggedStruct("PrivateAcceptance", { origin: AcceptanceOrigin }),
)
export type ArtifactDelivery = typeof ArtifactDelivery.Type
export const githubArtifactDelivery: ArtifactDelivery = { _tag: "Github" }

export const acceptsArtifactUrl = (policy: ArtifactDelivery, input: string): boolean => {
  if (policy._tag === "Github") return isGithubReleaseAssetUrl(input)
  if (!Schema.is(AcceptanceOrigin)(policy.origin)) return false
  try {
    const url = new URL(input)
    return url.origin === policy.origin && url.protocol === "https:" && !url.username && !url.password
      && !url.hash && url.href === input
  } catch { return false }
}

export const artifactDeliveryClient = (policy: ArtifactDelivery = githubArtifactDelivery) => {
  if (policy._tag === "Github") return githubDownloadClient
  return Effect.gen(function* () {
    const transport = (yield* HttpClient.HttpClient).pipe(HttpClient.withTracerDisabledWhen(() => true))
    return HttpClient.make(request => Effect.gen(function* () {
      const rejected = () => new HttpClientError.RequestError({ request, reason: "InvalidUrl", description: "Unexpected private acceptance artifact destination" })
      if (!acceptsArtifactUrl(policy, request.url)) return yield* rejected()
      const headers = Object.fromEntries(Object.entries(request.headers).filter(([name]) => ["range", "if-range", "accept-encoding"].includes(name)))
      const response = yield* transport.execute(HttpClientRequest.get(request.url, { headers }))
      if (response.status >= 300 && response.status < 400) return yield* rejected()
      return response
    }).pipe(Effect.provideService(FetchHttpClient.RequestInit, { redirect: "manual", credentials: "omit" })))
  })
}
