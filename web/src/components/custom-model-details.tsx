import { Option } from "effect"
import type { CustomLocalModel, LocalModel } from "@magnitudedev/sdk"

/** Where a custom model's files are: an added repository pinned to its commit, or a path Magnitude references. */
export const customModelSource = (model: CustomLocalModel): { readonly label: string; readonly href: Option.Option<string> } => {
  const origin = model.origin
  if (origin._tag === "Added") return { label: `${origin.repository} · pinned to ${origin.commit.slice(0, 7)}`, href: Option.some(`https://huggingface.co/${origin.repository}`) }
  const state = model.acquisitionState
  const path = (state._tag === "External" || state._tag === "Unavailable") && state.installation._tag === "Resolved" ? ` · ${state.installation.primaryPath}` : ""
  return { label: `${model.presentation.sourceLabel}${path}`, href: Option.none() }
}

/** A newer revision of the custom model's attached catalog drafter exists, and nothing is downloading it right now. */
export const drafterUpdateAvailable = (model: LocalModel): boolean =>
  model._tag === "Custom" && Option.exists(model.drafter, drafter => drafter.updateAvailable && drafter.availability._tag !== "Downloading")
