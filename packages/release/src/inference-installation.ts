import { Option } from "effect"
import { ICN_EXECUTABLE_NAME } from "./executables"
import type { ReleaseHost } from "./targets"

/** The planner inputs every inference installation carries at the same path. */
export const INFERENCE_PLANNER_BUNDLE = "catalog/model-planner-inputs.bundle"

/** The only top-level directories of an inference installation; there is no `backends/`. */
export const INFERENCE_INSTALLATION_DIRECTORIES = ["bin", "runtime", "catalog"] as const

export const inferenceExecutablePath = (host: ReleaseHost): string =>
  `bin/${ICN_EXECUTABLE_NAME}${host.executableExtension}`

/** Files NVIDIA's NVRTC redistributable contributes to `runtime/` on CUDA hosts. */
export const inferenceNvrtcPaths = (host: ReleaseHost): readonly string[] =>
  Option.match(host.nvrtc, {
    onNone: () => [],
    onSome: (nvrtc) => [...nvrtc.libraries, nvrtc.license].map((file) => `runtime/${file.name}`),
  })

/** Files every inference installation of the host must contain. */
export const inferenceRequiredPaths = (host: ReleaseHost): readonly string[] => [
  inferenceExecutablePath(host),
  INFERENCE_PLANNER_BUNDLE,
  ...inferenceNvrtcPaths(host),
]

export const isInferenceInstallationPath = (path: string): boolean =>
  INFERENCE_INSTALLATION_DIRECTORIES.some((directory) => path.startsWith(`${directory}/`))
