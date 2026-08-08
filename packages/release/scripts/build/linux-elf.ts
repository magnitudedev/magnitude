import { mkdir, mkdtemp, open, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { basename, resolve } from "node:path"
import type { HostId } from "../../src/targets"
import { fileSha256, run } from "./common"

const PLATFORM_LIBRARIES = new Set([
  "libc.so.6",
  "libdl.so.2",
  "libgcc_s.so.1",
  "libm.so.6",
  "libpthread.so.0",
  "libresolv.so.2",
  "librt.so.1",
  "libstdc++.so.6",
  "libutil.so.1",
])

const EXPECTED_MACHINE: Readonly<Record<string, string>> = {
  "linux-x64-gnu": "Advanced Micro Devices X86-64",
  "linux-arm64-gnu": "AArch64",
}

const EXPECTED_INTERPRETER: Readonly<Record<string, string>> = {
  "linux-x64-gnu": "/lib64/ld-linux-x86-64.so.2",
  "linux-arm64-gnu": "/lib/ld-linux-aarch64.so.1",
}

interface InspectedElf {
  readonly path: string
  readonly name: string
  readonly soname: string | undefined
  readonly needed: readonly string[]
  readonly digest: string
}

const files = async (root: string): Promise<readonly string[]> => {
  const found: string[] = []
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = resolve(directory, entry.name)
      if (entry.isDirectory()) await visit(path)
      else if (entry.isFile()) found.push(path)
    }
  }
  await visit(root)
  return found
}

const isElf = async (path: string): Promise<boolean> => {
  const file = await open(path, "r")
  try {
    const header = Buffer.alloc(4)
    const { bytesRead } = await file.read(header, 0, header.length, 0)
    return bytesRead === 4 && header.equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46]))
  } finally {
    await file.close()
  }
}

const compareVersion = (left: string, right: string): number => {
  const leftParts = left.split(".").map(Number)
  const rightParts = right.split(".").map(Number)
  for (let index = 0; index < Math.max(leftParts.length, rightParts.length); index++) {
    const difference = (leftParts[index] ?? 0) - (rightParts[index] ?? 0)
    if (difference !== 0) return difference
  }
  return 0
}

const matches = (text: string, expression: RegExp): readonly string[] =>
  Array.from(text.matchAll(expression), (match) => match[1]!).filter(
    (value, index, values) => values.indexOf(value) === index
  )

const inspectElf = async (host: HostId, path: string): Promise<InspectedElf> => {
  const report = await run([
    "readelf",
    "--file-header",
    "--program-headers",
    "--dynamic",
    "--version-info",
    path,
  ])
  const machine = report.match(/^\s*Machine:\s*(.+)$/m)?.[1]?.trim()
  const elfClass = report.match(/^\s*Class:\s*(.+)$/m)?.[1]?.trim()
  if (elfClass !== "ELF64") {
    throw new Error(`${path} has unexpected ELF class ${elfClass ?? "unknown"}`)
  }
  if (machine !== EXPECTED_MACHINE[host]) {
    throw new Error(
      `${path} targets ${machine ?? "an unknown architecture"}, expected ${EXPECTED_MACHINE[host]}`
    )
  }
  const interpreter = report.match(/Requesting program interpreter:\s*([^\]]+)/)?.[1]?.trim()
  if (interpreter !== undefined && interpreter !== EXPECTED_INTERPRETER[host]) {
    throw new Error(`${path} uses unexpected interpreter ${interpreter}`)
  }
  const loaderPaths = matches(report, /\((?:RUNPATH|RPATH)\).*?\[([^\]]+)\]/g).flatMap((value) =>
    value.split(":")
  )
  for (const loaderPath of loaderPaths) {
    if (loaderPath !== "$ORIGIN" && loaderPath !== "$ORIGIN/../runtime") {
      throw new Error(`${path} has non-owned loader path ${loaderPath}`)
    }
  }
  for (const version of matches(report, /\bGLIBC_(\d+\.\d+)\b/g)) {
    if (compareVersion(version, "2.35") > 0) {
      throw new Error(`${path} requires GLIBC_${version}; maximum is GLIBC_2.35`)
    }
  }
  for (const version of matches(report, /\bGLIBCXX_(\d+\.\d+\.\d+)\b/g)) {
    if (compareVersion(version, "3.4.30") > 0) {
      throw new Error(`${path} requires GLIBCXX_${version}; maximum is GLIBCXX_3.4.30`)
    }
  }
  return {
    path,
    name: basename(path),
    soname: report.match(/\(SONAME\).*?\[([^\]]+)\]/)?.[1],
    needed: matches(report, /\(NEEDED\).*?\[([^\]]+)\]/g),
    digest: await fileSha256(path),
  }
}

/** Validates the final Linux archive composition. */
export const verifyLinuxElfComposition = async (
  host: HostId,
  archives: readonly string[],
  capabilities: readonly string[] = []
): Promise<void> => {
  if (!host.startsWith("linux-")) return
  const root = await mkdtemp(resolve(tmpdir(), `magnitude-elf-${host}-`))
  try {
    const extracted = await Promise.all(
      archives.map(async (archive, index) => {
        const directory = resolve(root, String(index))
        await mkdir(directory, { recursive: true, mode: 0o700 })
        await run(["tar", "-xzf", archive, "-C", directory])
        return directory
      })
    )
    const elfPaths = (await Promise.all(extracted.map(files)))
      .flat()
      .filter((path, index, values) => values.indexOf(path) === index)
    const elfs = await Promise.all(
      (await Promise.all(elfPaths.map(async (path) => ((await isElf(path)) ? path : undefined))))
        .filter((path): path is string => path !== undefined)
        .map((path) => inspectElf(host, path))
    )
    const providers = new Map<string, InspectedElf>()
    for (const elf of elfs) {
      for (const name of [elf.name, elf.soname].filter(
        (value): value is string => value !== undefined
      )) {
        const existing = providers.get(name)
        if (existing !== undefined && existing.digest !== elf.digest) {
          throw new Error(
            `${name} is supplied by conflicting files ${existing.path} and ${elf.path}`
          )
        }
        providers.set(name, elf)
      }
    }
    const allowedExternalLibraries = new Set([...PLATFORM_LIBRARIES, ...capabilities])
    for (const elf of elfs) {
      for (const needed of elf.needed) {
        if (!providers.has(needed) && !allowedExternalLibraries.has(needed)) {
          throw new Error(`${elf.path} requires undeclared library ${needed}`)
        }
      }
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}
