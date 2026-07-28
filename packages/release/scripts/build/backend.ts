import { access, mkdir, rm } from "node:fs/promises"
import { basename, resolve } from "node:path"
import { Option } from "effect"
import {
  backendArchive,
  backendPacks,
  hostById,
} from "../../src/targets"
import { buildArchive, run, type ArchiveSource } from "./common"
import { buildIcnBinary } from "../../../../inference/scripts/compile"

const fixedCudaDirectories = (): readonly string[] => {
  const root = process.env.CUDA_PATH?.trim()
  return root
    ? [
      resolve(root, "bin"),
      resolve(root, "lib64"),
      resolve(root, "lib", "x64"),
    ]
    : []
}

const resolveRuntimeLibrary = async (
  name: string,
  buildFiles: readonly string[],
): Promise<string> => {
  const fromBuild = buildFiles.find((file) => basename(file) === name)
  if (fromBuild) return fromBuild
  for (const directory of fixedCudaDirectories()) {
    const candidate = resolve(directory, name)
    try {
      await access(candidate)
      return candidate
    } catch {
      // Try the next fixed toolkit directory.
    }
  }
  throw new Error(`backend build did not produce required runtime library ${name}`)
}

export const buildBackendArtifact = async (
  packId: string,
  outputRoot: string,
): Promise<void> => {
  const pack = backendPacks.find((candidate) => candidate.id === packId)
  if (!pack) throw new Error(`unknown backend pack ${packId}`)
  const host = hostById(pack.host)
  const output = resolve(outputRoot)
  await rm(output, { recursive: true, force: true })
  await mkdir(output, { recursive: true, mode: 0o700 })

  const icn = await buildIcnBinary({
    target: host.bunTarget,
    profile: `backend-${pack.id}`,
    features: pack.cargoFeatures,
    buildEnvironment: pack.backend === "cuda"
      ? {
        CMAKE_CUDA_ARCHITECTURES: pack.cudaArchitectures?.join(";") ?? "",
      }
      : {},
  })
  if (!icn.identity.backends.includes(pack.backend)) {
    throw new Error(`${pack.id} build identity does not include ${pack.backend}`)
  }
  const modules = icn.backendModules.filter(
    (file) => basename(file) === pack.module,
  )
  if (modules.length !== 1) {
    throw new Error(`${pack.id} emitted ${modules.length} ${pack.module} modules`)
  }
  const runtime = await Promise.all(
    pack.runtimeLibraries.map((name) =>
      resolveRuntimeLibrary(name, icn.runtimeLibraries)
    ),
  )
  if (host.id.startsWith("darwin-")) {
    for (const file of [...modules, ...runtime]) {
      await run(["codesign", "--force", "--sign", "-", file])
    }
  }
  const sources: ArchiveSource[] = [
    ...modules.map((source) => ({
      path: `backends/${basename(source)}`,
      source,
      mode: 0o755,
    })),
    ...runtime.map((source, index) => ({
      path: `runtime/${pack.runtimeLibraries[index]!}`,
      source,
      mode: 0o755,
    })),
  ]
  await buildArchive(
    resolve(output, backendArchive(pack)),
    resolve(output, `icn-backend-${pack.id}.artifact.json`),
    {
      id: `icn-backend-${pack.id}`,
      kind: "icn-backend",
      host: Option.some(pack.host),
      backend: Option.some(pack.backend),
      requiredBaseId: Option.some(`icn-base-${pack.host}`),
      nativeBuild: Option.some(icn.identity.native_build),
      backendModuleAbi: Option.some(icn.identity.backend_module_abi),
      compatibility: Option.some(pack.compatibility),
    },
    sources,
  )
}

if (import.meta.main) {
  const packId = process.argv[2]
  if (!packId) throw new Error("usage: backend.ts <pack-id> <output-root>")
  await buildBackendArtifact(
    packId,
    resolve(process.argv[3] ?? `release/${packId}`),
  )
}
