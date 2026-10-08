import { Option } from "effect"
import { linuxPackageArchitecture, linuxPackageVersion, type LinuxPackageFormat } from "./linux-package"

export type HostId =
  | "darwin-arm64"
  | "darwin-x64"
  | "linux-arm64-gnu"
  | "linux-x64-gnu"
  | "windows-x64-msvc"

export const MACOS_DEPLOYMENT_TARGET = "15.0" as const

/** One file taken from NVIDIA's NVRTC archive and its name in the installation's `runtime/`. */
export interface NvrtcFile {
  readonly member: string
  readonly name: string
}

/**
 * NVIDIA's pinned NVRTC redistributable: the entire CUDA payload of an inference artifact.
 * Only the two standard libraries ship (never the `.alt` variants), with NVIDIA's license notice.
 */
export interface NvrtcRedistributable {
  readonly version: string
  readonly url: string
  readonly bytes: number
  readonly sha256: string
  readonly libraries: readonly [NvrtcFile, NvrtcFile]
  readonly license: NvrtcFile
}

export interface ReleaseHost {
  readonly id: HostId
  readonly runner: string
  readonly bunTarget: string
  readonly rustTarget: string
  readonly executableExtension: "" | ".exe"
  readonly nvrtc: Option.Option<NvrtcRedistributable>
}

const NVRTC_VERSION = "12.9.86"
const NVRTC_REDISTRIBUTABLE = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc"
const NVRTC_LICENSE_NAME = "NVRTC-LICENSE.txt"

const linuxNvrtc = (platform: "linux-x86_64" | "linux-sbsa", bytes: number, sha256: string): NvrtcRedistributable => {
  const root = `cuda_nvrtc-${platform}-${NVRTC_VERSION}-archive`
  return {
    version: NVRTC_VERSION,
    url: `${NVRTC_REDISTRIBUTABLE}/${platform}/${root}.tar.xz`,
    bytes,
    sha256,
    libraries: [
      { member: `${root}/lib/libnvrtc.so.${NVRTC_VERSION}`, name: "libnvrtc.so.12" },
      { member: `${root}/lib/libnvrtc-builtins.so.${NVRTC_VERSION}`, name: "libnvrtc-builtins.so.12.9" },
    ],
    license: { member: `${root}/LICENSE`, name: NVRTC_LICENSE_NAME },
  }
}

const windowsNvrtcRoot = `cuda_nvrtc-windows-x86_64-${NVRTC_VERSION}-archive`

// This is product configuration, not a serialized registry or extension point.
export const releaseHosts = [
  {
    id: "darwin-arm64",
    runner: "blacksmith-12vcpu-macos-15",
    bunTarget: "bun-darwin-arm64",
    rustTarget: "aarch64-apple-darwin",
    executableExtension: "",
    nvrtc: Option.none(),
  },
  {
    id: "darwin-x64",
    runner: "macos-15-large",
    bunTarget: "bun-darwin-x64",
    rustTarget: "x86_64-apple-darwin",
    executableExtension: "",
    nvrtc: Option.none(),
  },
  {
    id: "linux-arm64-gnu",
    runner: "blacksmith-16vcpu-ubuntu-2204-arm",
    bunTarget: "bun-linux-arm64",
    rustTarget: "aarch64-unknown-linux-gnu",
    executableExtension: "",
    // Server-class arm64 (SBSA), including GB10; NVIDIA's linux-aarch64 archive is for Jetson.
    nvrtc: Option.some(linuxNvrtc("linux-sbsa", 53_265_740, "fb2d50c791465f333fc2236d2419170cf7a7886f48dd9b967a10f8233c686029")),
  },
  {
    id: "linux-x64-gnu",
    runner: "blacksmith-16vcpu-ubuntu-2204",
    bunTarget: "bun-linux-x64-baseline",
    rustTarget: "x86_64-unknown-linux-gnu",
    executableExtension: "",
    nvrtc: Option.some(linuxNvrtc("linux-x86_64", 114_231_976, "82913658363892dbc0f2638b070476234476e06e084fed60db861cb7e161a6af")),
  },
  {
    id: "windows-x64-msvc",
    runner: "blacksmith-16vcpu-windows-2025",
    bunTarget: "bun-windows-x64",
    rustTarget: "x86_64-pc-windows-msvc",
    executableExtension: ".exe",
    nvrtc: Option.some({
      version: NVRTC_VERSION,
      url: `${NVRTC_REDISTRIBUTABLE}/windows-x86_64/${windowsNvrtcRoot}.zip`,
      bytes: 314_608_748,
      sha256: "1aa0644fa53c8ca34cdc73db17bcc73530557bdd3f582c7bfdbd7916c8b48f65",
      libraries: [
        { member: `${windowsNvrtcRoot}/bin/nvrtc64_120_0.dll`, name: "nvrtc64_120_0.dll" },
        { member: `${windowsNvrtcRoot}/bin/nvrtc-builtins64_129.dll`, name: "nvrtc-builtins64_129.dll" },
      ],
      license: { member: `${windowsNvrtcRoot}/LICENSE`, name: NVRTC_LICENSE_NAME },
    }),
  },
] as const satisfies readonly ReleaseHost[]

export const hostById = (id: HostId): ReleaseHost => {
  const host = releaseHosts.find((candidate) => candidate.id === id)
  if (!host) throw new Error(`Unknown release host ${id}`)
  return host
}

export const releaseBuildEnvironment = (
  host: ReleaseHost,
): Readonly<Record<string, string>> =>
  host.id.startsWith("darwin-")
    ? {
      MACOSX_DEPLOYMENT_TARGET: MACOS_DEPLOYMENT_TARGET,
      CMAKE_OSX_DEPLOYMENT_TARGET: MACOS_DEPLOYMENT_TARGET,
    }
    : {}

export const cliArchive = (host: HostId) => `magnitude-cli-${host}.tar.gz`
export const acnArchive = (host: HostId) => `magnitude-acn-${host}.tar.gz`
export const desktopInstaller = (host: "darwin-arm64" | "darwin-x64") => `magnitude-desktop-${host}.dmg`
export const desktopUpdateArchive = (host: "darwin-arm64" | "darwin-x64") => `magnitude-desktop-${host}.zip`
export const windowsDesktopInstaller = (version: string) => `magnitude-desktop-windows-x64-${version}.exe`
export const linuxDesktopInstaller = (host: "linux-arm64-gnu" | "linux-x64-gnu", format: LinuxPackageFormat, version: string, revision: number) => {
  const arch = linuxPackageArchitecture(format, host === "linux-arm64-gnu" ? "arm64" : "x64")
  switch (format) {
    case "deb": return `magnitude-desktop_${version}-${revision}_${arch}.deb`
    case "rpm": return `magnitude-desktop-${version}-${revision}.${arch}.rpm`
    case "pacman": return `magnitude-desktop-${linuxPackageVersion(format, version)}-${revision}-${arch}.pkg.tar.zst`
  }
}
export const icnBaseArchive = (host: HostId) => `magnitude-icn-base-${host}.tar.gz`

export const currentHost = (): HostId => {
  const key = `${process.platform}-${process.arch}`
  if (key === "darwin-arm64") return "darwin-arm64"
  if (key === "darwin-x64") return "darwin-x64"
  if (key === "linux-arm64") return "linux-arm64-gnu"
  if (key === "linux-x64") return "linux-x64-gnu"
  if (key === "win32-x64") return "windows-x64-msvc"
  throw new Error(`Unsupported release host ${key}`)
}
