import type {
  ArtifactInstallationEvent,
  ReleaseBundleSizes,
} from "@magnitudedev/release"
import type { IcnStartupBackend } from "@magnitudedev/icn-protocol"
import { Context, Effect } from "effect"

export type IcnPreparationBackend =
  | { readonly _tag: "Cpu"; readonly hardwareLabel: string }
  | { readonly _tag: "Metal"; readonly hardwareLabel: string }
  | { readonly _tag: "Cuda"; readonly hardwareLabel: string }
  | { readonly _tag: "Vulkan"; readonly hardwareLabel: string }

export const icnPreparationBackend = (
  backend: IcnStartupBackend,
): IcnPreparationBackend => {
  switch (backend.type) {
    case "cpu": return { _tag: "Cpu", hardwareLabel: backend.hardwareLabel }
    case "metal": return { _tag: "Metal", hardwareLabel: backend.hardwareLabel }
    case "cuda": return { _tag: "Cuda", hardwareLabel: backend.hardwareLabel }
    case "vulkan": return { _tag: "Vulkan", hardwareLabel: backend.hardwareLabel }
  }
}

export type IcnPreparationEvent =
  | { readonly _tag: "Resolving" }
  | {
      readonly _tag: "Planned"
      readonly plan: ReleaseBundleSizes
    }
  | { readonly _tag: "InstallationRequired" }
  | {
      readonly _tag: "Artifact"
      readonly artifact: "Base" | "Accelerator"
      readonly event: ArtifactInstallationEvent
    }
  | { readonly _tag: "Starting" }
  | {
      readonly _tag: "PreparingBackend"
      readonly backend: IcnPreparationBackend
    }

export interface IcnPreparationReporter {
  readonly report: (event: IcnPreparationEvent) => Effect.Effect<void>
}

export const IcnPreparationReporter =
  Context.GenericTag<IcnPreparationReporter>(
    "@magnitudedev/icn/IcnPreparationReporter",
  )
