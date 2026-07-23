export type IcnRuntimeBackend = "cpu" | "cuda"

interface IcnReleaseHost {
  readonly platform: NodeJS.Platform
  readonly arch: string
  readonly requestedBackend: string | undefined
  readonly nvidiaDriverAvailable: boolean
}

const basePlatformKey = (
  platform: NodeJS.Platform,
  arch: string,
): string => {
  if (platform === "darwin" && arch === "arm64") return "darwin-arm64"
  if (platform === "darwin" && arch === "x64") return "darwin-x64"
  if (platform === "linux" && arch === "x64") return "linux-x64"
  if (platform === "linux" && arch === "arm64") return "linux-arm64"
  if (platform === "win32" && arch === "x64") return "windows-x64"
  throw new Error(`Unsupported ICN platform: ${platform} ${arch}`)
}

export const selectIcnReleaseBackend = (
  host: IcnReleaseHost,
): IcnRuntimeBackend => {
  const requested = host.requestedBackend?.trim().toLowerCase() || "auto"
  if (!["auto", "cpu", "cuda"].includes(requested)) {
    throw new Error(
      `Unsupported MAGNITUDE_ICN_BACKEND=${requested}; expected auto, cpu, or cuda`,
    )
  }
  if (requested === "cuda") {
    if (host.platform !== "linux") {
      throw new Error("The CUDA ICN release backend is supported only on Linux")
    }
    if (!host.nvidiaDriverAvailable) {
      throw new Error("CUDA was requested, but no NVIDIA driver is available")
    }
    return "cuda"
  }
  if (requested === "cpu") return "cpu"
  return host.platform === "linux" && host.nvidiaDriverAvailable ? "cuda" : "cpu"
}

export const selectIcnReleasePlatformKey = (
  host: IcnReleaseHost,
): string => {
  const base = basePlatformKey(host.platform, host.arch)
  return selectIcnReleaseBackend(host) === "cuda" ? `${base}-cuda` : base
}
