import { homedir } from "node:os"
import { installedAcquisition, type LocalModel } from "@magnitudedev/sdk"

const targetPackage = (model: LocalModel) =>
  model.bundle._tag === "Standalone" ? model.bundle.package : model.bundle.target

export const discoveredModelLocation = (model: LocalModel): string => {
  const acquisition = installedAcquisition(model.acquisitionState)
  if (acquisition === undefined) {
    throw new Error("Discovered model summary requires an installed model")
  }
  const targetId = targetPackage(model).id
  const installed = acquisition.packages.find(({ packageId }) => packageId === targetId)
  if (installed === undefined) {
    throw new Error(`Installed model is missing target package location ${targetId}`)
  }
  const home = homedir()
  const separator = installed.path[home.length]
  return installed.path === home
    ? "~"
    : installed.path.startsWith(home) && (separator === "/" || separator === "\\")
      ? `~${installed.path.slice(home.length)}`
      : installed.path
}
