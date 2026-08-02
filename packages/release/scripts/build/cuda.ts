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

const sortedImages = (
  images: ReadonlyMap<string, CudaPtxImage>,
): readonly CudaPtxImage[] =>
  [...images.values()].sort((left, right) =>
    left.target - right.target
      || Number(left.architectureSpecific) - Number(right.architectureSpecific)
      || left.ptxVersion.localeCompare(right.ptxVersion))

const ptxImageInspector = () => {
  const images = new Map<string, CudaPtxImage>()
  let version: string | undefined
  const accept = (line: string): void => {
    const nextVersion = line.match(/\.version\s+(\d+\.\d+)/)?.[1]
    if (nextVersion) {
      version = nextVersion
      return
    }
    const target = line.match(/\.target\s+sm_(\d+)(a)?\b/)
    if (!version || !target) return
    const image = {
      ptxVersion: version,
      target: Number(target[1]),
      architectureSpecific: target[2] === "a",
    }
    images.set(imageKey(image), image)
    version = undefined
  }
  return { accept, images: () => sortedImages(images) }
}

export const inspectPtxImages = (dump: string): readonly CudaPtxImage[] => {
  const inspector = ptxImageInspector()
  for (const line of dump.split("\n")) inspector.accept(line)
  return inspector.images()
}

const inspectPtxImagesFromModule = async (
  cuobjdump: string,
  module: string,
): Promise<readonly CudaPtxImage[]> => {
  const child = Bun.spawn([cuobjdump, "--dump-ptx", module], {
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  })
  const inspector = ptxImageInspector()
  const readOutput = async (): Promise<void> => {
    const reader = child.stdout.getReader()
    const decoder = new TextDecoder()
    let pending = ""
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const lines = `${pending}${decoder.decode(value, { stream: true })}`.split("\n")
      pending = lines.pop() ?? ""
      for (const line of lines) inspector.accept(line)
    }
    pending += decoder.decode()
    if (pending.length > 0) inspector.accept(pending)
  }
  const [code, , stderr] = await Promise.all([
    child.exited,
    readOutput(),
    new Response(child.stderr).text(),
  ])
  if (code !== 0) {
    throw new Error(
      `${cuobjdump} failed with exit ${code}: ${stderr.trim().slice(0, 4_000)}`,
    )
  }
  return inspector.images()
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
  const [images, compilerOutput] = await Promise.all([
    inspectPtxImagesFromModule(resolve(cudaRoot, "bin", "cuobjdump"), module),
    run([resolve(cudaRoot, "bin", "nvcc"), "--version"]),
  ])
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
