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

export interface BackendPack {
  readonly id: string
  readonly host: HostId
  readonly backend: Exclude<Backend, "cpu">
  readonly runner: string
  readonly cargoFeatures: readonly string[]
  readonly module: string
  readonly runtimeLibraries: readonly string[]
  readonly cudaArchitectures?: readonly string[]
  readonly compatibility:
    | {
      readonly kind: "metal"
    }
    | {
      readonly kind: "cuda"
      readonly toolkit: string
      readonly minimumDriverApi: number
      readonly minimumArchitecture: number
    }
    | {
      readonly kind: "vulkan"
      readonly minimumApi: string
    }
}

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

// Windows release artifacts are intentionally disabled for now. Runtime support outside the
// release system remains available to revisit once Windows builds are reliable.
export const backendPacks = [
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
  {
    id: "cuda13-linux-arm64-gnu",
    host: "linux-arm64-gnu",
    backend: "cuda",
    runner: "ubuntu-24.04-arm",
    cargoFeatures: ["dynamic-backends", "cuda-no-vmm"],
    module: "libggml-cuda.so",
    runtimeLibraries: ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"],
    // PTX-only tiers keep release builds manageable while retaining the CUDA code paths that
    // matter most today: Ampere+, Hopper, and Blackwell. Additional compatibility tiers or native
    // cubins can be restored later if measured startup or runtime performance justifies the cost.
    cudaArchitectures: ["80-virtual", "90-virtual", "120-virtual"],
    compatibility: {
      kind: "cuda",
      toolkit: "13.0",
      minimumDriverApi: 13000,
      minimumArchitecture: 80,
    },
  },
  {
    id: "cuda12-linux-x64-gnu",
    host: "linux-x64-gnu",
    backend: "cuda",
    runner: "ubuntu-latest",
    cargoFeatures: ["dynamic-backends", "cuda-no-vmm"],
    module: "libggml-cuda.so",
    runtimeLibraries: ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"],
    // Keep this list aligned with the ARM64 pack above.
    cudaArchitectures: ["80-virtual", "90-virtual", "120-virtual"],
    compatibility: {
      kind: "cuda",
      toolkit: "12.9",
      minimumDriverApi: 12000,
      minimumArchitecture: 80,
    },
  },
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
] as const satisfies readonly BackendPack[]

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
