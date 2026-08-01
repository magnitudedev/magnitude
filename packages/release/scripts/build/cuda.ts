import { resolve } from "node:path"
import { run } from "./common"

export interface CudaPtxImage {
  readonly ptxVersion: string
  readonly target: number
  readonly architectureSpecific: boolean
}

const DRIVER_API_BY_PTX_VERSION: Readonly<Record<string, number>> = {
  "7.8": 11080,
  "8.8": 12090,
}

const imageKey = (image: CudaPtxImage): string =>
  `${image.ptxVersion}:${image.target}${image.architectureSpecific ? "a" : ""}`

export const inspectPtxImages = (dump: string): readonly CudaPtxImage[] => {
  const images = new Map<string, CudaPtxImage>()
  const modules = dump.split(/(?=\.version\s+\d+\.\d+)/g)
  for (const module of modules) {
    const version = module.match(/\.version\s+(\d+\.\d+)/)?.[1]
    const target = module.match(/\.target\s+sm_(\d+)(a)?\b/)
    if (!version || !target) continue
    const image = {
      ptxVersion: version,
      target: Number(target[1]),
      architectureSpecific: target[2] === "a",
    }
    images.set(imageKey(image), image)
  }
  return [...images.values()].sort((left, right) =>
    left.target - right.target
      || Number(left.architectureSpecific) - Number(right.architectureSpecific)
      || left.ptxVersion.localeCompare(right.ptxVersion))
}

export const inspectNvccCompiler = (output: string): string => {
  const identity = output
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.startsWith("Cuda compilation tools, release "))
  if (!identity) throw new Error("nvcc did not report its compiler identity")
  return identity
}

export const inspectCudaCompatibility = async (
  module: string,
  configuration: {
    readonly toolkitVersion: string
  },
) => {
  const cudaRoot = process.env.CUDA_PATH?.trim()
  if (!cudaRoot) throw new Error("CUDA_PATH is required to inspect a CUDA pack")
  const [dump, compilerOutput] = await Promise.all([
    run([resolve(cudaRoot, "bin", "cuobjdump"), "--dump-ptx", module]),
    run([resolve(cudaRoot, "bin", "nvcc"), "--version"]),
  ])
  const images = inspectPtxImages(dump)
  const [firstImage] = images
  if (!firstImage) {
    throw new Error("finished CUDA module contains no inspectable PTX images")
  }
  const imagesWithDriverFloors = images.map((image) => {
    const floor = DRIVER_API_BY_PTX_VERSION[image.ptxVersion]
    if (floor === undefined) {
      throw new Error(`PTX ${image.ptxVersion} has no reviewed driver-JIT floor`)
    }
    return { ...image, minimumDriverApi: floor }
  })
  const [firstCompatibleImage, ...remainingCompatibleImages] = imagesWithDriverFloors
  if (!firstCompatibleImage) throw new Error("finished CUDA module contains no compatible PTX images")
  const compiler = inspectNvccCompiler(compilerOutput)
  if (!compiler.includes(`release ${configuration.toolkitVersion}`)) {
    throw new Error(
      `nvcc identity ${JSON.stringify(compiler)} does not match configured CUDA ${configuration.toolkitVersion}`,
    )
  }
  return {
    kind: "cuda" as const,
    toolkitVersion: configuration.toolkitVersion,
    compiler,
    images: [firstCompatibleImage, ...remainingCompatibleImages] as const,
  }
}
