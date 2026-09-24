import { FetchHttpClient } from "@effect/platform"
import { Config, Effect, Option } from "effect"
import { resolveReleaseIcnInstallation } from "../../../icn/src/lifecycle/release-installation"
import { IcnPreparationReporter } from "../../../icn/src/lifecycle/preparation"
import { releaseBaseUrl } from "../../src/acquisition"

/** Update fixture versions are unpublished; service readiness uses an explicit real engine. */
export const acceptanceInferenceInstallation = (dataDirectory: string) => Effect.gen(function* () {
  const supplied = yield* Config.option(Config.string("MAGNITUDE_ICN_PATH"))
  if (Option.isSome(supplied)) return supplied.value
  const version = yield* Config.string("MAGNITUDE_ACCEPTANCE_ICN_VERSION")
  const installation = yield* resolveReleaseIcnInstallation(version, dataDirectory, releaseBaseUrl()).pipe(
    Effect.provideService(IcnPreparationReporter, { report: () => Effect.void }), Effect.provide(FetchHttpClient.layer))
  return installation.declarationPath
})
