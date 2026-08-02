export type HostId =
  | "darwin-arm64"
  | "darwin-x64"
  | "linux-arm64-gnu"
  | "linux-x64-gnu"
  | "windows-x64-msvc"

export type Backend = "cpu" | "metal" | "cuda" | "vulkan"

export interface ReleaseHost {
  readonly id: HostId
  readonly runner: string
  readonly bunTarget: string
  readonly rustTarget: string
  readonly executableExtension: "" | ".exe"
  readonly cargoFeatures: readonly string[]
}

interface BackendPackBase {
  readonly id: string
  readonly host: HostId
  readonly runner: string
  readonly cargoFeatures: readonly string[]
  readonly module: string
  readonly runtimeLibraries: readonly string[]
}

export type BackendPack =
  | (BackendPackBase & {
      readonly backend: "cuda"
      readonly cuda: {
        readonly toolkitVersion: string
        readonly architectures: readonly string[]
      }
    })
  | (BackendPackBase & {
      readonly backend: "metal"
      readonly compatibility: {
        readonly kind: "metal"
      }
    })
  | (BackendPackBase & {
      readonly backend: "vulkan"
      readonly compatibility: {
        readonly kind: "vulkan"
        readonly minimumApi: string
      }
    })

// This is product configuration, not a serialized registry or extension point.
export const releaseHosts = [
  {
    id: "darwin-arm64",
    runner: "macos-latest",
    bunTarget: "bun-darwin-arm64",
    rustTarget: "aarch64-apple-darwin",
    executableExtension: "",
    cargoFeatures: ["mtmd", "dynamic-backends"],
  },
  {
    id: "darwin-x64",
    runner: "macos-15-intel",
    bunTarget: "bun-darwin-x64",
    rustTarget: "x86_64-apple-darwin",
    executableExtension: "",
    cargoFeatures: ["mtmd", "dynamic-backends"],
  },
  {
    id: "linux-arm64-gnu",
    runner: "ubuntu-24.04-arm",
    bunTarget: "bun-linux-arm64",
    rustTarget: "aarch64-unknown-linux-gnu",
    executableExtension: "",
    cargoFeatures: ["mtmd", "dynamic-backends"],
  },
  {
    id: "linux-x64-gnu",
    runner: "ubuntu-latest",
    bunTarget: "bun-linux-x64-baseline",
    rustTarget: "x86_64-unknown-linux-gnu",
    executableExtension: "",
    cargoFeatures: ["mtmd", "dynamic-backends"],
  },
] as const satisfies readonly ReleaseHost[]

const cudaBuilds = [
  {
    toolkitVersion: "11.8",
    architectures: ["80-virtual"],
    runtimeLibraries: ["libcudart.so.11.0", "libcublas.so.11", "libcublasLt.so.11"],
  },
  {
    toolkitVersion: "12.9",
    architectures: ["80-virtual", "90-virtual", "120-virtual"],
    runtimeLibraries: ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"],
  },
] as const

const cudaHosts = [
  {
    host: "linux-arm64-gnu",
    runners: { "11.8": "ubuntu-24.04-arm", "12.9": "ubuntu-24.04-arm" },
  },
  {
    host: "linux-x64-gnu",
    runners: { "11.8": "ubuntu-24.04", "12.9": "ubuntu-24.04" },
  },
] as const

const cudaBackendPacks: readonly BackendPack[] = cudaHosts.flatMap(({ host, runners }) =>
  cudaBuilds.map((cuda) => ({
    id: `cuda-${cuda.toolkitVersion}-${host}`,
    host,
    backend: "cuda" as const,
    runner: runners[cuda.toolkitVersion],
    cargoFeatures: ["dynamic-backends", "cuda-no-vmm"],
    module: "libggml-cuda.so",
    runtimeLibraries: cuda.runtimeLibraries,
    cuda,
  })))

// Windows release artifacts are intentionally disabled for now. Runtime support outside the
// release system remains available to revisit once Windows builds are reliable.
export const backendPacks: readonly BackendPack[] = [
  {
    id: "metal-darwin-arm64",
    host: "darwin-arm64",
    backend: "metal",
    runner: "macos-latest",
    cargoFeatures: ["dynamic-backends", "metal"],
    module: "libggml-metal.so",
    runtimeLibraries: [],
    compatibility: { kind: "metal" },
  },
  ...cudaBackendPacks,
  {
    id: "vulkan1-linux-arm64-gnu",
    host: "linux-arm64-gnu",
    backend: "vulkan",
    runner: "ubuntu-24.04-arm",
    cargoFeatures: ["dynamic-backends", "vulkan"],
    module: "libggml-vulkan.so",
    runtimeLibraries: [],
    compatibility: { kind: "vulkan", minimumApi: "1.1.0" },
  },
  {
    id: "vulkan1-linux-x64-gnu",
    host: "linux-x64-gnu",
    backend: "vulkan",
    runner: "ubuntu-latest",
    cargoFeatures: ["dynamic-backends", "vulkan"],
    module: "libggml-vulkan.so",
    runtimeLibraries: [],
    compatibility: { kind: "vulkan", minimumApi: "1.1.0" },
  },
]

export const hostById = (id: HostId): ReleaseHost => {
  const host = releaseHosts.find((candidate) => candidate.id === id)
  if (!host) throw new Error(`Unknown release host ${id}`)
  return host
}

export const cliArchive = (host: HostId) => `magnitude-cli-${host}.tar.gz`
export const acnArchive = (host: HostId) => `magnitude-acn-${host}.tar.gz`
export const icnBaseArchive = (host: HostId) => `magnitude-icn-base-${host}.tar.gz`
export const backendArchive = (pack: BackendPack) => `magnitude-icn-${pack.id}.tar.gz`

export const currentHost = (): HostId => {
  const key = `${process.platform}-${process.arch}`
  if (key === "darwin-arm64") return "darwin-arm64"
  if (key === "darwin-x64") return "darwin-x64"
  if (key === "linux-arm64") return "linux-arm64-gnu"
  if (key === "linux-x64") return "linux-x64-gnu"
  if (key === "win32-x64") return "windows-x64-msvc"
  throw new Error(`Unsupported release host ${key}`)
}
