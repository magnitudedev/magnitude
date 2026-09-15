import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { basename, join } from "node:path"

class WindowsRuntimeInvalid extends Schema.TaggedError<WindowsRuntimeInvalid>()("WindowsRuntimeInvalid", {
  message: Schema.String,
}) {}

// Windows provides these DLLs and API sets. The MSVC redistributable is deliberately
// absent: it must be carried by the installation, even when installed on the builder.
const systemLibraries = new Set([
  "advapi32.dll", "avrt.dll", "bcrypt.dll", "bcryptprimitives.dll", "cfgmgr32.dll",
  "combase.dll", "comctl32.dll", "comdlg32.dll", "crypt32.dll", "dbghelp.dll", "dnsapi.dll",
  "gdi32.dll", "imm32.dll", "iphlpapi.dll", "kernel32.dll", "mswsock.dll",
  "ncrypt.dll", "netapi32.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll",
  "pdh.dll", "powrprof.dll", "propsys.dll", "psapi.dll", "rpcrt4.dll", "secur32.dll",
  "setupapi.dll", "shell32.dll", "shlwapi.dll", "synchronization.dll", "ucrtbase.dll",
  "user32.dll", "userenv.dll", "version.dll", "winhttp.dll", "winmm.dll",
  "wintrust.dll", "ws2_32.dll", "wtsapi32.dll",
])

export const windowsSystemLibrary = (name: string): boolean => {
  const normalized = name.toLowerCase()
  return systemLibraries.has(normalized) || /^(?:api|ext)-ms-win-[a-z0-9-]+\.dll$/.test(normalized)
}

export const windowsImportedLibraries = (report: string): readonly string[] =>
  [...new Set([...report.matchAll(/^\s+([\w.-]+\.dll)\s*$/gim)].map(match => match[1]!.toLowerCase()))]

/** Close the native import graph using only owned files, the selected MSVC CRT, and Windows. */
export const collectWindowsRuntime = (input: {
  readonly files: readonly string[]
  readonly redistributable: string
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const available = new Map<string, string>()
  for (const name of yield* fs.readDirectory(input.redistributable)) {
    if (name.toLowerCase().endsWith(".dll")) available.set(name.toLowerCase(), join(input.redistributable, name))
  }
  const owned = new Map<string, string>()
  for (const file of input.files) {
    const name = basename(file).toLowerCase()
    if (owned.has(name)) return yield* new WindowsRuntimeInvalid({ message: `Duplicate Windows runtime filename: ${name}` })
    owned.set(name, file)
  }
  const additional: string[] = []
  const pending = [...input.files]
  for (let index = 0; index < pending.length; index++) {
    const file = pending[index]!
    const report = yield* Command.make("dumpbin.exe", "/dependents", file).pipe(Command.string)
    const imports = windowsImportedLibraries(report)
    if (imports.length === 0) return yield* new WindowsRuntimeInvalid({ message: `No Windows imports reported for ${file}` })
    for (const dependency of imports) {
      if (owned.has(dependency) || windowsSystemLibrary(dependency)) continue
      const source = available.get(dependency)
      if (!source) return yield* new WindowsRuntimeInvalid({ message: `${basename(file)} requires unbundled Windows library ${dependency}` })
      owned.set(dependency, source)
      pending.push(source)
      additional.push(source)
    }
  }
  return additional.sort()
}).pipe(Effect.provide(BunContext.layer))
