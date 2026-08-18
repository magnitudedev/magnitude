import type { ModelParameterization, ModelReleaseDate } from "@magnitudedev/sdk"

const MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1_000

const formatParameterCount = (parameters: number): string =>
  `${Math.round(parameters / 1_000_000_000)}B`

export const formatModelClassification = (
  parameterization: ModelParameterization,
  vision: boolean,
): string => {
  const modality = vision ? "Text and vision" : "Text only"
  return parameterization.architecture === "dense"
    ? `Dense (${formatParameterCount(parameterization.totalParameters)}) · ${modality}`
    : `MoE (${formatParameterCount(parameterization.totalParameters)} / ${formatParameterCount(parameterization.activeParameters)}) · ${modality}`
}

export const formatModelReleaseRecency = (
  releaseDate: ModelReleaseDate,
  now = new Date(),
): string => {
  const [year, month, day] = releaseDate.split("-").map(Number) as [number, number, number]
  const releaseDay = Date.UTC(year, month - 1, day)
  const currentDay = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate())
  const elapsedDays = Math.max(0, Math.round((currentDay - releaseDay) / MILLISECONDS_PER_DAY))
  return elapsedDays === 1 ? "1 day ago" : `${elapsedDays} days ago`
}
