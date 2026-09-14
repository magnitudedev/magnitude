import { FetchHttpClient, HttpClient, HttpClientError, HttpClientRequest } from "@effect/platform"
import { Effect, Stream } from "effect"
import { isGithubDeliveryUrl, isGithubReleaseAssetUrl } from "./github-artifact"

/** Every hop is admitted before transport. Only byte-transfer headers leave this boundary. */
export const githubDownloadClient = Effect.gen(function* () {
  const transport = (yield* HttpClient.HttpClient).pipe(HttpClient.withTracerDisabledWhen(() => true))
  return HttpClient.make(request => Effect.gen(function* () {
    const rejected = () => new HttpClientError.RequestError({ request, reason: "InvalidUrl", description: "Unexpected release delivery destination" })
    if (!isGithubReleaseAssetUrl(request.url)) return yield* rejected()
    const headers = Object.fromEntries(Object.entries(request.headers).filter(([name]) => ["range", "if-range", "accept-encoding"].includes(name)))
    let location = request.url
    for (let hop = 0; hop < 5; hop++) {
      if (!isGithubDeliveryUrl(location)) return yield* rejected()
      const response = yield* transport.execute(HttpClientRequest.get(location, { headers }))
      if (![301, 302, 303, 307, 308].includes(response.status)) return response
      yield* response.stream.pipe(Stream.take(1), Stream.runDrain, Effect.ignore)
      const next = response.headers.location
      if (!next) return yield* rejected()
      location = next
    }
    return yield* rejected()
  }).pipe(Effect.provideService(FetchHttpClient.RequestInit, { redirect: "manual", credentials: "omit" })))
})
