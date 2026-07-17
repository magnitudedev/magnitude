import { Option } from "effect"
import type { LocalInferenceLlamaCppInstallation, LocalInferenceState } from "@magnitudedev/sdk"

export type LlamaCppInstallationStatus = "ready" | "outdated" | "missing"

export interface LlamaCppInstallationManagementView {
  readonly status: LlamaCppInstallationStatus
  readonly minimumBuild: number
  readonly recommendedBuild: number
  readonly installations: readonly LocalInferenceLlamaCppInstallation[]
  readonly selected: Option.Option<LocalInferenceLlamaCppInstallation>
  readonly active: Option.Option<LocalInferenceLlamaCppInstallation>
  readonly representativeOutdated: Option.Option<LocalInferenceLlamaCppInstallation>
  readonly managedInstall: LocalInferenceState["llamaCpp"]["managedInstall"]
  readonly managedInstallRecommended: boolean
}

export interface LlamaCppInstallationChatNotice {
  readonly kind: "outdated" | "missing"
  readonly prefix: string
  readonly actionLabel: "/settings"
  readonly suffix: string
}

const discoveryPriority = (installation: LocalInferenceLlamaCppInstallation): number => Math.min(
  ...installation.discoveries.map((discovery) => discovery._tag === "Configured"
    ? 0
    : discovery._tag === "Managed"
      ? 1
      : 2 + discovery.priority),
)

export const deriveLlamaCppInstallationManagementView = (
  state: LocalInferenceState,
): LlamaCppInstallationManagementView => {
  const llama = state.llamaCpp
  const selected = Option.flatMap(llama.selectedInstallationId, (id) =>
    Option.fromNullable(llama.installations.find((installation) => installation.id === id)),
  )
  const active = Option.flatMap(llama.activeManagedInstallationId, (id) =>
    Option.fromNullable(llama.installations.find((installation) => installation.id === id)),
  )
  const representativeOutdated = Option.fromNullable(llama.installations.toSorted((left, right) =>
    right.build - left.build
    || discoveryPriority(left) - discoveryPriority(right)
    || left.executables.serverPath.localeCompare(right.executables.serverPath),
  )[0])
  return {
    status: Option.isSome(selected) ? "ready" : llama.installations.length > 0 ? "outdated" : "missing",
    minimumBuild: llama.minimumBuild,
    recommendedBuild: llama.recommendedBuild,
    installations: llama.installations,
    selected,
    active,
    representativeOutdated,
    managedInstall: llama.managedInstall,
    managedInstallRecommended: !llama.installations.some((installation) =>
      installation.ownership === "magnitude" && installation.build >= llama.recommendedBuild),
  }
}

export const deriveLlamaCppInstallationChatNotice = (
  state: LocalInferenceState,
  localDemand: boolean,
): Option.Option<LlamaCppInstallationChatNotice> => {
  const view = deriveLlamaCppInstallationManagementView(state)
  if (view.status === "ready") return Option.none()
  if (view.status === "outdated") return Option.map(view.representativeOutdated, (installation) => ({
    kind: "outdated",
    prefix: `llama.cpp b${installation.build} is outdated. Magnitude requires b${view.minimumBuild} or newer. Open `,
    actionLabel: "/settings",
    suffix: ` to install b${view.recommendedBuild}.`,
  }))
  return localDemand
    ? Option.some({
        kind: "missing",
        prefix: "llama.cpp is not installed for local inference. Open ",
        actionLabel: "/settings",
        suffix: ` to install b${view.recommendedBuild}.`,
      })
    : Option.none()
}
