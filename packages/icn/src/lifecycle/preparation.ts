import type {
  ArtifactInstallationEvent,
  ReleaseBundleSizes,
} from "@magnitudedev/release"
import { Context, Effect } from "effect"

export type IcnPreparationEvent =
  | { readonly _tag: "Resolving" }
  | {
      readonly _tag: "Planned"
      readonly plan: ReleaseBundleSizes
    }
  | { readonly _tag: "InstallationRequired" }
  | {
      readonly _tag: "Artifact"
      readonly event: ArtifactInstallationEvent
    }
  | { readonly _tag: "Starting" }
  | {
      readonly _tag: "Retrying"
      readonly message: string
      readonly retryAtMillis: number
    }

export interface IcnPreparationReporter {
  readonly report: (event: IcnPreparationEvent) => Effect.Effect<void>
}

export const IcnPreparationReporter =
  Context.GenericTag<IcnPreparationReporter>(
    "@magnitudedev/icn/IcnPreparationReporter",
  )
