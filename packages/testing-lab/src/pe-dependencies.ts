import { Effect, Option, Schema } from "effect"
import { win32 } from "node:path"
import { AssertionFailure } from "./domain"
import { checkedCommand } from "./process"

const moduleName = Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9_.-]+\.(dll|drv)$/i))
const ImportSymbol = Schema.Struct({ Name: Schema.NullOr(Schema.String), ModuleName: moduleName,
  ImportByOrdinal: Schema.Boolean, Ordinal: Schema.Int, DelayImport: Schema.Boolean })
const ImportReport = Schema.Struct({ Imports: Schema.Array(Schema.Struct({ Name: moduleName,
  NumberOfEntries: Schema.Int, ImportList: Schema.Array(ImportSymbol) })) })
export const PeImport = Schema.Struct({ name: moduleName, linkage: Schema.Literal("required", "delay") })
const PeResolution = Schema.Struct({ ModuleName: moduleName, Filepath: Schema.NullOr(Schema.NonEmptyString), SearchStrategy: Schema.Int })
const PeModule = Schema.Struct({ Filepath: Schema.NonEmptyString, Imports: ImportReport.fields.Imports,
  Dependencies: Schema.Array(PeResolution) })
export type PeModule = typeof PeModule.Type
const GraphReport = Schema.Struct({ schemaVersion: Schema.Literal(1), Root: Schema.NonEmptyString, Modules: Schema.Array(PeModule) })
const fail = (message: string) => new AssertionFailure({ message: `PE dependency inspection: ${message}` })
export const pePathKey = (path: string) => win32.normalize(path).toLowerCase()

/** Dependencies v1.11's native import table includes both ordinary and delayed imports. */
export const decodePeImports = (json: string) => Schema.decodeUnknown(Schema.parseJson(ImportReport))(json).pipe(
  Effect.mapError(() => fail("malformed import report")), Effect.flatMap(report => verifyPeImports(report.Imports)))

export const verifyPeImports = (table: typeof ImportReport.Type["Imports"]) => Effect.gen(function* () {
  const imports: (typeof PeImport.Type)[] = []
  for (const module of table) {
    if (module.NumberOfEntries < 1 || module.NumberOfEntries !== module.ImportList.length) return yield* fail("incomplete import table")
    const linkages = new Set<"required" | "delay">()
    for (const symbol of module.ImportList) {
      if (symbol.ModuleName.toLowerCase() !== module.Name.toLowerCase()
        || (symbol.ImportByOrdinal ? symbol.Ordinal < 0 || symbol.Ordinal > 65535 : !symbol.Name)) return yield* fail("inconsistent import symbol")
      linkages.add(symbol.DelayImport ? "delay" : "required")
    }
    for (const linkage of linkages) imports.push({ name: module.Name.toLowerCase(), linkage })
  }
  return imports
})

/** Preserve the root executable's search context; do not resolve a child under a new root. */
export const decodePeGraph = (json: string, executable: string) => Effect.gen(function* () {
  const report = yield* Schema.decodeUnknown(Schema.parseJson(GraphReport))(json).pipe(
    Effect.mapError(() => fail("malformed loader report")))
  if (pePathKey(report.Root) !== pePathKey(executable)) return yield* fail("loader report has a different root")
  const modules = new Map<string, PeModule>()
  if (!report.Modules.length || report.Modules.length > 16_384) return yield* fail("invalid owned graph size")
  for (const module of report.Modules) {
    const key = pePathKey(module.Filepath)
    if (modules.has(key)) return yield* fail("duplicate loader context")
    for (const path of [module.Filepath, ...module.Dependencies.flatMap(dependency => dependency.Filepath === null ? [] : [dependency.Filepath])]) {
      if (!win32.isAbsolute(path) || path.startsWith("\\\\")) return yield* fail("loader returned a nonlocal path")
    }
    modules.set(key, module)
  }
  if (!modules.has(pePathKey(executable))) return yield* fail("loader report omitted its root")
  return modules
})

/** The pinned library resolves imports in one root context. Traversal stops at OS boundaries. */
export const peInspectionScript = String.raw`$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$directory = Split-Path -Parent $env:LAB_PE_INSPECTOR
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class LabPeNative { [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] public static extern bool SetDllDirectory(string path); }'
if (-not [LabPeNative]::SetDllDirectory($directory)) { throw 'Cannot load the pinned native inspector' }
[void][Reflection.Assembly]::LoadFrom((Join-Path $directory 'ClrPhlib.dll'))
[void][Reflection.Assembly]::LoadFrom((Join-Path $directory 'DependenciesLib.dll'))
[void][Dependencies.ClrPh.Phlib]::InitializePhLib()
[Dependencies.BinaryCache]::InitializeBinaryCache($false)
$root = [Dependencies.BinaryCache]::LoadPe($env:LAB_PE_FILE)
if (-not $root -or -not $root.LoadSuccessful) { throw 'Native root is not a valid PE' }
$prefix = [IO.Path]::GetFullPath($env:LAB_PE_ROOT).TrimEnd('\') + '\'
$queue = [Collections.Generic.Queue[string]]::new()
$queue.Enqueue($root.Filepath)
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$modules = [Collections.Generic.List[object]]::new()
$search = [Collections.Generic.List[string]]::new()
$sxs = [Dependencies.SxsManifest]::GetSxsEntries($root)
while ($queue.Count) {
  if ($seen.Count + $queue.Count -gt 16384) { throw 'Owned PE graph exceeds inspection limit' }
  $file = $queue.Dequeue()
  if (-not $seen.Add($file)) { continue }
  if (-not $file.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Native root escapes the owned directory' }
  $pe = [Dependencies.BinaryCache]::LoadPe($file)
  if (-not $pe -or -not $pe.LoadSuccessful) { throw 'Owned dependency is not a valid PE' }
  $imports = @($pe.GetImports())
  $dependencies = [Collections.Generic.List[object]]::new()
  foreach ($item in $imports) {
    $resolved = [Dependencies.BinaryCache]::ResolveModule($root, $item.Name, $sxs, $search, $env:LAB_PE_WORKING_DIRECTORY)
    $path = if ($resolved.Item2) { $resolved.Item2.Filepath } else { $null }
    $dependencies.Add(@{ModuleName=$item.Name;Filepath=$path;SearchStrategy=[int]$resolved.Item1})
    if ($path -and $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { $queue.Enqueue($path) }
  }
  $modules.Add(@{Filepath=$pe.Filepath;Imports=$imports;Dependencies=@($dependencies.ToArray())})
}
@{schemaVersion=1;Root=$root.Filepath;Modules=@($modules.ToArray())} | ConvertTo-Json -Depth 8 -Compress
`

export const inspectPeGraph = (tool: string, file: string, root: string, workingDirectory: string, systemRoot: string, ownedSearchPaths: readonly string[]) =>
  checkedCommand(win32.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
    ["-NoProfile", "-NonInteractive", "-Command", peInspectionScript], {
      cwd: Option.some(workingDirectory), inheritEnv: false,
      env: { SystemRoot: systemRoot, WINDIR: systemRoot, PATH: [...ownedSearchPaths, `${systemRoot}\\System32`, systemRoot].join(";"),
        TEMP: workingDirectory, TMP: workingDirectory,
        LAB_PE_INSPECTOR: tool, LAB_PE_FILE: file, LAB_PE_ROOT: root, LAB_PE_WORKING_DIRECTORY: workingDirectory },
      timeoutMs: 60_000, maxOutputBytes: 16 * 1024 * 1024,
    }).pipe(Effect.flatMap(result => decodePeGraph(result.stdout, file)))
