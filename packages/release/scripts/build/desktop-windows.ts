import * as Command from "@effect/platform/Command"
import type { PlatformError } from "@effect/platform/Error"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Schema } from "effect"
import { valid } from "semver"
import { basename, dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { DesktopBuildFailed } from "./desktop"
import { ReleaseArtifactSchema } from "../../src/contracts"
import { sha256File } from "../../src/macos-app"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..")
const validPayloadPath = (path: string) => path.split("/").every(segment =>
  segment.length > 0 && !/[<>:"\\|?*\x00-\x1f\x7f]/.test(segment) &&
  !/[. ]$/.test(segment) && !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(segment),
)
export const WindowsPayloadPath = Schema.String.pipe(
  Schema.filter(validPayloadPath, { message: () => "Expected a relative Windows payload filename" }),
  Schema.brand("WindowsPayloadPath"),
)
export const WindowsInstallerInput = Schema.Struct({
  version: Schema.String.pipe(Schema.filter(value => valid(value) !== null && /^[0-9]/.test(value) && value.trim() === value, { message: () => "Expected a canonical release version" })),
  revision: Schema.Int.pipe(Schema.between(0, 65535)),
  files: Schema.NonEmptyArray(WindowsPayloadPath),
})
const quote = (value: string) => value.replaceAll("$", () => "$$").replaceAll('"', '$\\"').replaceAll("'", "$\\'")
const remove = (path: string, directory: boolean) =>
  `  System::Call '$PLUGINSDIR\\MagnitudeInstallGuard.dll::RemovePayload(w "${quote(path.replaceAll("/", "\\"))}", i ${Number(directory)}) i .r0'\n  \${If} $0 != 0\n    Goto removalFailed\n  \${EndIf}`

const prepareWindowsInstallerInput = (input: typeof WindowsInstallerInput.Encoded) => Effect.gen(function* () {
  const options = yield* Schema.decodeUnknown(WindowsInstallerInput)(input)
  const files = [...options.files].sort()
  const names = new Set<string>()
  const directories = new Set<string>(["resources"])
  for (const file of files) {
    const key = file.toUpperCase()
    if (names.has(key) || key.split("/")[0] === "UNINSTALL MAGNITUDE.EXE" || key === "RESOURCES/INSTALLATION-FILES.TXT") {
      return yield* new DesktopBuildFailed({ message: `Conflicting Windows payload filename: ${file}` })
    }
    names.add(key)
    const segments = file.split("/")
    segments.pop()
    while (segments.length) { directories.add(segments.join("/")); segments.pop() }
  }
  const directoryNames = new Map<string, string>()
  for (const directory of directories) {
    const prior = directoryNames.get(directory.toUpperCase())
    if (prior !== undefined && prior !== directory) return yield* new DesktopBuildFailed({ message: `Windows payload directory casing collision: ${directory}` })
    directoryNames.set(directory.toUpperCase(), directory)
    if (names.has(directory.toUpperCase())) return yield* new DesktopBuildFailed({ message: `Windows payload file/directory collision: ${directory}` })
  }
  const versionParts = options.version.split(/[+-]/)[0]!.split(".").map(Number)
  if (versionParts.some(part => part > 65535)) return yield* new DesktopBuildFailed({ message: "Windows installer version components exceed the native version range" })
  return { options, files, directories, versionParts }
})

/** The inventory describes only files owned by this installer, including its generated records. */
export const renderWindowsInstallationInventory = (input: typeof WindowsInstallerInput.Encoded) => prepareWindowsInstallerInput(input).pipe(
  Effect.map(({ options, files, directories }) => [
    "magnitude-installation-v1", options.version,
    ...[...files, "resources/installation-files.txt", "Uninstall Magnitude.exe"].sort().map(file => `F\t${file.replaceAll("/", "\\")}`),
    ...[...directories].sort((a, b) => b.split("/").length - a.split("/").length || a.localeCompare(b)).map(directory => `D\t${directory.replaceAll("/", "\\")}`),
    "",
  ].join("\n")),
)

/** One validated file set governs extraction, removal and the installed inventory. */
export const renderWindowsInstaller = (template: string, input: typeof WindowsInstallerInput.Encoded) => Effect.gen(function* () {
  const { options, files, directories, versionParts } = yield* prepareWindowsInstallerInput(input)
  const prefix = `!define MAGNITUDE_VERSION "${quote(options.version)}"`
  const replacements: Record<string, string> = {
    "@PAYLOAD_FILES@": files.map((file, index) => {
      const parent = file.includes("/") ? "\\" + file.slice(0, file.lastIndexOf("/")).replaceAll("/", "\\") : ""
      return `  SetOutPath "$Stage${quote(parent)}"\n  IfErrors stageFailed\n  File "/oname=${quote(file.split("/").at(-1)!)}" "payload\\${String(index).padStart(6, "0")}.bin"\n  IfErrors stageFailed`
    }).join("\n"),
    "@REMOVE_FILES@": [...files, "resources/installation-files.txt"].map(file => remove(file, false)).join("\n"),
    "@REMOVE_DIRECTORIES@": [...directories].sort((a, b) => b.split("/").length - a.split("/").length || a.localeCompare(b)).map(directory => remove(directory, true)).join("\n"),
  }
  for (const token of Object.keys(replacements)) {
    if (template.split(token).length !== 2) return yield* new DesktopBuildFailed({ message: `Installer template must contain exactly one ${token}` })
  }
  const metadata = `VIProductVersion "${versionParts.join(".")}.${options.revision}"\nVIAddVersionKey "ProductName" "Magnitude"\nVIAddVersionKey "CompanyName" "Magnitude"\nVIAddVersionKey "FileDescription" "Magnitude Setup"\nVIAddVersionKey "FileVersion" "${quote(options.version)}"\nVIAddVersionKey "ProductVersion" "${quote(options.version)}"`
  return `${prefix}\n${metadata}\n${template.replace(/@(?:PAYLOAD_FILES|REMOVE_FILES|REMOVE_DIRECTORIES)@/g, token => replacements[token]!)}`
})

/** Package a matched application with an explicitly built x86 NSIS helper. No publication or signing claim. */
export const buildWindowsDesktopInstaller = (options: {
  readonly app: string; readonly guard: string; readonly makensis: string
  readonly version: string; readonly revision: number; readonly output: string
}) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const stage = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-windows-installer-" })
  const source = yield* fs.realPath(resolve(options.app))
  const files: string[] = []
  const inspect = (directory: string): Effect.Effect<void, PlatformError | DesktopBuildFailed> => Effect.gen(function* () {
    for (const name of yield* fs.readDirectory(join(source, directory))) {
      const relative = directory ? `${directory}/${name}` : name
      yield* Schema.decodeUnknown(WindowsPayloadPath)(relative).pipe(Effect.mapError(() => new DesktopBuildFailed({ message: `Invalid Windows payload filename: ${relative}` })))
      const path = join(source, ...relative.split("/"))
      if ((yield* fs.realPath(path)) !== path) return yield* new DesktopBuildFailed({ message: `Windows payload contains a redirected path: ${relative}` })
      const info = yield* fs.stat(path)
      if (info.type === "Directory") yield* inspect(relative)
      else if (info.type === "File") files.push(relative)
      else return yield* new DesktopBuildFailed({ message: `Unsupported Windows payload entry: ${relative}` })
    }
  })
  yield* inspect("")
  for (const required of ["Magnitude.exe", "resources/app.asar", "resources/magnitude-service.exe", "resources/desktop-host.node", "resources/Magnitude-LICENSE.txt"]) {
    if (!files.includes(required)) return yield* new DesktopBuildFailed({ message: `Windows application is missing ${required}` })
  }
  const guard = yield* fs.readFile(resolve(options.guard))
  const pe = guard.length >= 64 ? new DataView(guard.buffer, guard.byteOffset, guard.byteLength).getUint32(60, true) : 0
  if (guard[0] !== 77 || guard[1] !== 90 || pe < 64 || pe + 24 > guard.length ||
      new DataView(guard.buffer, guard.byteOffset, guard.byteLength).getUint32(pe, true) !== 0x4550 ||
      new DataView(guard.buffer, guard.byteOffset, guard.byteLength).getUint16(pe + 4, true) !== 0x14c ||
      !(new DataView(guard.buffer, guard.byteOffset, guard.byteLength).getUint16(pe + 22, true) & 0x2000)) {
    return yield* new DesktopBuildFailed({ message: "NSIS installer helper must be an x86 PE DLL" })
  }
  const app = join(stage, "payload")
  files.sort()
  for (const [index, file] of files.entries()) {
    const destination = join(app, `${String(index).padStart(6, "0")}.bin`)
    yield* fs.makeDirectory(dirname(destination), { recursive: true })
    yield* fs.copyFile(join(source, ...file.split("/")), destination)
  }
  const helper = join(stage, "MagnitudeInstallGuard.dll")
  yield* fs.writeFile(helper, guard)
  const candidate = join(stage, "Magnitude-setup.exe")
  const template = yield* fs.readFileString(join(root, "packages/release/resources/windows/desktop.nsi"))
  yield* fs.copyFile(join(root, "packages/release/resources/windows/Magnitude.ico"), join(stage, "Magnitude.ico"))
  const script = yield* renderWindowsInstaller(template, { version: options.version, revision: options.revision, files: files as [string, ...string[]] })
  const inventory = yield* renderWindowsInstallationInventory({ version: options.version, revision: options.revision, files: files as [string, ...string[]] })
  yield* fs.writeFile(join(stage, "installation-files.txt"), Buffer.from(`\uFEFF${inventory}`, "utf16le"))
  const scriptPath = join(stage, "desktop.nsi")
  yield* fs.writeFileString(scriptPath, script)
  const code = yield* Command.make(options.makensis, scriptPath).pipe(Command.workingDirectory(stage), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
  if (code !== 0) return yield* new DesktopBuildFailed({ message: `Windows installer compilation exited ${code}` })
  if ((yield* fs.stat(candidate)).size === 0n) return yield* new DesktopBuildFailed({ message: "Windows installer compilation produced an empty artifact" })
  yield* fs.makeDirectory(options.output, { recursive: true })
  const output = resolve(options.output, `magnitude-desktop-windows-x64-${options.version}.exe`)
  yield* fs.copyFile(candidate, output)
  const artifact = yield* Schema.decodeUnknown(ReleaseArtifactSchema)({
    id: "desktop-windows-x64-msvc", kind: "desktop", host: "windows-x64-msvc",
    filename: basename(output), bytes: Number((yield* fs.stat(output)).size),
    sha256: yield* sha256File(output),
  })
  yield* fs.writeFileString(join(options.output, `${artifact.id}.artifact.json`),
    yield* Schema.encode(Schema.parseJson(ReleaseArtifactSchema))(artifact), { flag: "wx", mode: 0o600 })
  return { output, artifact }
})).pipe(Effect.mapError(error => error instanceof DesktopBuildFailed ? error : new DesktopBuildFailed({ message: String(error) })))
