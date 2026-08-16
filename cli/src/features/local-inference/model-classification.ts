import type { ModelParameterization } from "@magnitudedev/sdk"

const formatParameterCount = (parameters: number): string =>
  `${Math.round(parameters / 1_000_000_000)}B`

export const formatModelClassification = (
  parameterization: ModelParameterization,
  vision: boolean,
): string => {
  const modality = vision ? "Vision and text" : "Text only"
  return parameterization.architecture === "dense"
    ? `Dense · ${formatParameterCount(parameterization.totalParameters)} total parameters · ${modality}`
    : `MoE · ${formatParameterCount(parameterization.totalParameters)} total / ${formatParameterCount(parameterization.activeParameters)} active parameters · ${modality}`
}
