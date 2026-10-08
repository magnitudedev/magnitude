import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Schema } from "effect"
import { basename, dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { DesktopBuildFailed } from "./desktop"
import { LINUX_DESKTOP_PACKAGE_NAME } from "../../src/executables"
import { ReleaseArtifactSchema } from "../../src/contracts"
import { sha256File } from "../../src/macos-app"
import { LinuxPackageFormat, linuxPackageArchitecture, linuxPackageVersion, pacmanPackageIdentity } from "../../src/linux-package"
import { linuxDesktopInstaller } from "../../src/targets"
import { renderLinuxMaintainerScripts, renderPacmanInstallScript } from "./linux-maintainer-scripts"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..")
const PackageFields = {
    name: Schema.Literal(LINUX_DESKTOP_PACKAGE_NAME),
    productName: Schema.Literal("Magnitude"),
    genericName: Schema.String,
    description: Schema.String,
    productDescription: Schema.String,
    maintainer: Schema.String,
    homepage: Schema.String,
    bin: Schema.Literal("launch"),
    revision: Schema.String,
    icon: Schema.Record({ key: Schema.String, value: Schema.String }),
    categories: Schema.Array(Schema.String),
    desktopTemplate: Schema.String,
    version: Schema.String,
}
const DebianOptions = Schema.Struct({ options: Schema.Struct({ ...PackageFields, compression: Schema.Literal("none"), depends: Schema.Array(Schema.String), scripts: Schema.Record({ key: Schema.String, value: Schema.String }) }) })
const RpmOptions = Schema.Struct({ options: Schema.Struct({ ...PackageFields, requires: Schema.Array(Schema.String), license: Schema.String, specTemplate: Schema.String }) })

const desktopEntry = `[Desktop Entry]\nName=Magnitude\nComment=Discover and run local models\nExec=${LINUX_DESKTOP_PACKAGE_NAME}\nIcon=${LINUX_DESKTOP_PACKAGE_NAME}\nType=Application\nTerminal=false\nStartupNotify=true\nStartupWMClass=${LINUX_DESKTOP_PACKAGE_NAME}\nCategories=Development;\n`
const packageDescription = "Discover and run local models"
const packageMaintainer = "Magnitude <founders@magnitude.dev>"
const packageHomepage = "https://magnitude.dev"
const applicationIcon = join(root, "assets/brand/application-icon.png")
const pacmanHookPath = "usr/share/libalpm/hooks/zz-magnitude-desktop.hook"
const pacmanBeginScriptPath = `usr/share/libalpm/scripts/${LINUX_DESKTOP_PACKAGE_NAME}-installation-begin`
// Libraries the payload links against, plus flock, pkexec, xdg-open and the updater's bsdtar.
const pacmanDepends = ["alsa-lib", "at-spi2-core", "cairo", "dbus", "expat", "glib2", "glibc", "gtk3", "libarchive", "libcups",
  "libgcc", "libnotify", "libsecret", "libx11", "libxcb", "libxcomposite", "libxdamage", "libxext", "libxfixes", "libxkbcommon",
  "libxrandr", "mesa", "nspr", "nss", "pango", "polkit", "systemd-libs", "util-linux", "xdg-utils"]

export const validateLinuxPayloadPermissions = (format: LinuxPackageFormat, listing: string) => Effect.gen(function* () {
  const prefix = `/usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}`
  const rows = listing.split("\n").filter(line => format === "deb" ? line.includes(` .${prefix}/`)
    : format === "pacman" ? line.includes(` ${prefix.slice(1)}/`)
    : line.startsWith(`${prefix}/`) || line.startsWith(`${prefix} `))
  if (rows.length === 0 || rows.some(line => {
    if (format === "deb") {
      const [mode, owner] = line.trim().split(/\s+/)
      return owner !== "root/root" || !mode || (mode[0] !== "l" && (mode[5] === "w" || mode[8] === "w"))
    }
    if (format === "pacman") {
      const [mode, , owner, group] = line.trim().split(/\s+/)
      return owner !== "root" || group !== "root" || !mode || (mode[0] !== "l" && (mode[5] === "w" || mode[8] === "w"))
    }
    const [, rawMode, owner, group] = line.split(" ")
    const mode = Number.parseInt(rawMode ?? "", 8)
    return owner !== "root" || group !== "root" || !Number.isFinite(mode)
      || ((mode & 0o170000) !== 0o120000 && (mode & 0o022) !== 0)
  })) return yield* new DesktopBuildFailed({ message: "Linux application payload must be root-owned without group or other write access" })
})

export const validateLinuxDesktopInstaller = (options: {
  readonly file: string
  readonly format: LinuxPackageFormat
  readonly arch: "arm64" | "x64"
  readonly version: string
  readonly revision: number
}) => Effect.gen(function* () {
  const arch = linuxPackageArchitecture(options.format, options.arch)
  const identity = options.format === "pacman"
    ? pacmanPackageIdentity(yield* Command.make("bsdtar", "-xOf", options.file, ".PKGINFO").pipe(Command.string))
    : yield* (options.format === "deb"
      ? Command.make("dpkg-deb", "--show", "--showformat=${Package}\t${Version}\t${Architecture}", options.file)
      : Command.make("rpm", "-qp", "--qf", "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}", options.file)).pipe(Command.string)
  const expected = `${LINUX_DESKTOP_PACKAGE_NAME}\t${linuxPackageVersion(options.format, options.version)}-${options.revision}\t${arch}`
  if (identity.trim() !== expected) return yield* new DesktopBuildFailed({ message: "Linux desktop package identity does not match its release target" })
  if (options.format === "pacman") {
    const listing = yield* Command.make("bsdtar", "-tvf", options.file).pipe(Command.env({ LC_ALL: "C" }), Command.string)
    yield* validateLinuxPayloadPermissions("pacman", listing)
    const entries = new Map(listing.split("\n").map(line => {
      const [mode, , owner, group] = line.trim().split(/\s+/)
      return [line.slice(line.lastIndexOf(" ") + 1), `${mode} ${owner} ${group}`] as const
    }))
    if (entries.get(`usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/chrome-sandbox`) !== "-rwsr-xr-x root root") {
      return yield* new DesktopBuildFailed({ message: "Pacman installer must contain a root-owned mode-04755 Chromium sandbox helper" })
    }
    if (![".PKGINFO", ".MTREE", ".INSTALL", pacmanHookPath, pacmanBeginScriptPath].every(path => entries.has(path))) {
      return yield* new DesktopBuildFailed({ message: "Pacman installer is missing its metadata or installation admission hook" })
    }
    return
  }
  if (options.format === "deb") {
    const listing = yield* Command.make("dpkg-deb", "--contents", options.file).pipe(Command.env({ LC_ALL: "C" }), Command.string)
    yield* validateLinuxPayloadPermissions("deb", listing)
    const sandbox = listing.split("\n").find(line => line.endsWith(` ./usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/chrome-sandbox`))
    if (sandbox === undefined || !/^-rwsr-xr-x\s+root\/root\s/.test(sandbox)) {
      return yield* new DesktopBuildFailed({ message: "Debian installer must contain a root-owned mode-04755 Chromium sandbox helper" })
    }
  } else {
    const listing = yield* Command.make("rpm", "-qp", "--qf", "[%{FILENAMES} %{FILEMODES:octal} %{FILEUSERNAME} %{FILEGROUPNAME}\n]", options.file).pipe(Command.string)
    yield* validateLinuxPayloadPermissions("rpm", listing)
    if (!listing.split("\n").includes(`/usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/chrome-sandbox 104755 root root`)) {
      return yield* new DesktopBuildFailed({ message: "RPM installer must contain a root-owned mode-04755 Chromium sandbox helper" })
    }
  }
})

interface LinuxInstallerOptions {
  readonly format: LinuxPackageFormat
  readonly version: string
  readonly app: string
  readonly output: string
  readonly arch: "arm64" | "x64"
  readonly revision: number
}

/** Dpkg and rpm packages are produced by electron-installer from the staged payload. */
const buildElectronInstaller = (options: LinuxInstallerOptions & { readonly format: "deb" | "rpm" }, stage: string, payload: string, begin: string, end: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const config = join(stage, "options.json")
  const scripts: Record<string, string> = {}
  for (const [name, body] of Object.entries(renderLinuxMaintainerScripts(begin, end))) {
    scripts[name] = join(stage, name)
    yield* fs.writeFileString(scripts[name]!, `#!/bin/sh\nset -eu\n${body}`)
    yield* fs.chmod(scripts[name]!, 0o755)
  }
  const specTemplate = join(stage, "desktop.spec.ejs")
  const spec = yield* fs.readFileString(join(root, "packages/release/resources/linux/desktop.spec.ejs"))
  yield* fs.writeFileString(specTemplate, spec.replaceAll("@MAGNITUDE_INSTALL_BEGIN@", begin).replaceAll("@MAGNITUDE_INSTALL_END@", end))
  const desktopTemplate = join(stage, "magnitude.desktop.ejs")
  yield* fs.writeFileString(desktopTemplate, desktopEntry)
  const metadata = {
    name: LINUX_DESKTOP_PACKAGE_NAME, productName: "Magnitude", genericName: "Local inference",
    description: packageDescription,
    productDescription: "Magnitude provides curated model discovery, local inference, and configuration for external agent harnesses.",
    maintainer: packageMaintainer, homepage: packageHomepage,
    bin: "launch", revision: String(options.revision),
    icon: { "256x256": applicationIcon },
    categories: ["Development"], desktopTemplate,
    version: linuxPackageVersion(options.format, options.version),
  } as const
  yield* fs.writeFileString(config, options.format === "deb"
    ? yield* Schema.encode(Schema.parseJson(DebianOptions))({ options: { ...metadata, compression: "none", depends: ["libc6 (>= 2.35)", "libasound2t64 | libasound2", "util-linux", "pkexec"], scripts } })
    : yield* Schema.encode(Schema.parseJson(RpmOptions))({ options: { ...metadata, requires: ["glibc >= 2.35", "alsa-lib", "util-linux", "polkit"], license: "Apache-2.0", specTemplate } }))
  const destination = join(stage, "packages")
  const tool = options.format === "deb" ? "electron-installer-debian" : "electron-installer-redhat"
  const cli = join(dirname(fileURLToPath(import.meta.resolve(tool))), "cli.js")
  // The installer uses fs.chmod for the setuid sandbox. Bun 1.3.14 drops those
  // special mode bits on Linux; run this native packaging boundary under Node.
  const code = yield* Command.make("node", cli, "--src", payload, "--dest", destination,
    "--arch", linuxPackageArchitecture(options.format, options.arch), "--config", config).pipe(
    Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
  )
  if (code !== 0) return yield* new DesktopBuildFailed({ message: `${options.format} desktop packaging exited ${code}` })
  const packages = (yield* fs.readDirectory(destination)).filter(path => path.endsWith(`.${options.format}`))
  if (packages.length !== 1) return yield* new DesktopBuildFailed({ message: `${options.format} desktop packaging did not produce exactly one installer` })
  const filename = linuxDesktopInstaller(options.arch === "arm64" ? "linux-arm64-gnu" : "linux-x64-gnu", options.format, options.version, options.revision)
  // Package metadata uses '~' for prerelease ordering; GitHub rewrites it in asset names.
  if (packages[0] !== filename.replace(options.version, metadata.version)) {
    return yield* new DesktopBuildFailed({ message: "Linux desktop installer filename does not match its release target" })
  }
  const candidate = join(destination, packages[0]!)
  if (options.format === "deb") {
    // Both public executables are package-owned; maintainer scripts never edit user PATHs.
    const contents = join(stage, "debian-contents")
    const extracted = yield* Command.make("dpkg-deb", "--raw-extract", candidate, contents).pipe(Command.exitCode)
    if (extracted !== 0) return yield* new DesktopBuildFailed({ message: "Could not prepare the bundled CLI package entry" })
    yield* fs.symlink(`../lib/${LINUX_DESKTOP_PACKAGE_NAME}/resources/magnitude`, join(contents, "usr/bin/magnitude"))
    // Copied Electron directories can retain the builder's group-writable mode.
    // Normalize the final package tree, including directories created by the installer.
    const application = join(contents, "usr/lib", LINUX_DESKTOP_PACKAGE_NAME)
    for (const [type, mode] of [["d", "0755"], ["f", "go-w"]] as const) {
      const normalized = yield* Command.make("find", application, "-type", type, "-exec", "chmod", mode, "{}", "+").pipe(Command.exitCode)
      if (normalized !== 0) return yield* new DesktopBuildFailed({ message: "Could not protect the Debian application payload" })
    }
    // Compress only the final package; Ubuntu's default zstd level 19 is needlessly slow.
    const rebuilt = yield* Command.make("dpkg-deb", "--root-owner-group", "-Zzstd", "-z9", "--build", contents, candidate).pipe(Command.exitCode)
    if (rebuilt !== 0) return yield* new DesktopBuildFailed({ message: "Could not package the bundled CLI entry" })
  }
  return candidate
})

/**
 * Pacman packages are a zstd tar of the installed tree with .PKGINFO, .INSTALL and a gzipped
 * mtree, assembled as makepkg does. Bsdtar records root ownership without fakeroot, so any Linux
 * build host can produce them.
 */
const buildPacmanPackage = (options: LinuxInstallerOptions, stage: string, payload: string, begin: string, end: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const tree = join(stage, "pacman")
  const run = (failure: string, ...args: readonly [string, ...string[]]) => Command.make(...args).pipe(Command.workingDirectory(tree),
    Command.env({ LC_ALL: "C" }), Command.stderr("inherit"), Command.exitCode,
    Effect.flatMap(code => code === 0 ? Effect.void : new DesktopBuildFailed({ message: failure })))
  const pkgver = `${linuxPackageVersion("pacman", options.version)}-${options.revision}`
  if (!/^[A-Za-z0-9._+]+-[1-9][0-9]*$/.test(pkgver)) return yield* new DesktopBuildFailed({ message: "Release version cannot be represented as a pacman package version" })
  yield* fs.makeDirectory(join(tree, "usr/lib"), { recursive: true })
  yield* fs.rename(payload, join(tree, "usr/lib", LINUX_DESKTOP_PACKAGE_NAME))
  yield* fs.makeDirectory(join(tree, "usr/bin"))
  yield* fs.symlink(`../lib/${LINUX_DESKTOP_PACKAGE_NAME}/launch`, join(tree, "usr/bin", LINUX_DESKTOP_PACKAGE_NAME))
  yield* fs.symlink(`../lib/${LINUX_DESKTOP_PACKAGE_NAME}/resources/magnitude`, join(tree, "usr/bin/magnitude"))
  yield* fs.makeDirectory(join(tree, "usr/share/applications"), { recursive: true })
  yield* fs.writeFileString(join(tree, "usr/share/applications", `${LINUX_DESKTOP_PACKAGE_NAME}.desktop`), desktopEntry)
  yield* fs.makeDirectory(join(tree, "usr/share/icons/hicolor/256x256/apps"), { recursive: true })
  yield* fs.copyFile(applicationIcon, join(tree, "usr/share/icons/hicolor/256x256/apps", `${LINUX_DESKTOP_PACKAGE_NAME}.png`))
  yield* fs.makeDirectory(join(tree, "usr/share/libalpm/hooks"), { recursive: true })
  yield* fs.copyFile(join(root, "packages/release/resources/linux/zz-magnitude-desktop.hook"), join(tree, pacmanHookPath))
  yield* fs.makeDirectory(join(tree, "usr/share/libalpm/scripts"))
  yield* fs.writeFileString(join(tree, pacmanBeginScriptPath), `#!/bin/sh\nset -eu\n${begin}`)
  yield* fs.writeFileString(join(tree, ".INSTALL"), renderPacmanInstallScript(begin, end))
  // Files keep the builder's umask; only the hook script and the sandbox helper need special modes.
  yield* run("Could not protect the pacman package tree", "find", "usr", "-type", "d", "-exec", "chmod", "0755", "{}", "+")
  yield* run("Could not protect the pacman package tree", "find", "usr", "-type", "f", "-exec", "chmod", "go-w", "{}", "+")
  yield* fs.chmod(join(tree, pacmanBeginScriptPath), 0o755)
  yield* fs.chmod(join(tree, "usr/lib", LINUX_DESKTOP_PACKAGE_NAME, "chrome-sandbox"), 0o4755)
  const size = yield* Command.make("du", "-sb", "usr").pipe(Command.workingDirectory(tree), Command.string)
  const info = [
    "# Generated by Magnitude release packaging",
    `pkgname = ${LINUX_DESKTOP_PACKAGE_NAME}`, `pkgbase = ${LINUX_DESKTOP_PACKAGE_NAME}`, "xdata = pkgtype=pkg",
    `pkgver = ${pkgver}`, `pkgdesc = ${packageDescription}`, `url = ${packageHomepage}`,
    `builddate = ${Math.floor(Date.now() / 1000)}`, `packager = ${packageMaintainer}`,
    `size = ${Number.parseInt(size, 10)}`, `arch = ${linuxPackageArchitecture("pacman", options.arch)}`, "license = Apache-2.0",
    ...pacmanDepends.map(name => `depend = ${name}`),
  ]
  yield* fs.writeFileString(join(tree, ".PKGINFO"), `${info.join("\n")}\n`)
  const owner = ["--uid", "0", "--gid", "0", "--uname", "root", "--gname", "root"] as const
  yield* run("Could not record the pacman package file manifest", "bsdtar", "-czf", ".MTREE", "--format=mtree",
    "--options=!all,use-set,type,uid,gid,mode,time,size,md5,sha256,link", ...owner, ".PKGINFO", ".INSTALL", "usr")
  const candidate = join(stage, linuxDesktopInstaller(options.arch === "arm64" ? "linux-arm64-gnu" : "linux-x64-gnu", "pacman", options.version, options.revision))
  yield* run("Could not create the pacman package", "bsdtar", "-cf", candidate, "--zstd", "--options=zstd:compression-level=19",
    ...owner, ".MTREE", ".PKGINFO", ".INSTALL", "usr")
  return candidate
})

/** Package an already assembled, matched app/service. Publication requires native consumer acceptance. */
export const buildLinuxDesktopInstaller = (options: LinuxInstallerOptions) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const stage = yield* fs.makeTempDirectoryScoped({ prefix: `magnitude-desktop-${options.format}-` })
  const payload = join(stage, "application")
  yield* fs.copy(resolve(options.app), payload)
  yield* fs.writeFileString(join(payload, "resources/update-package.json"), yield* Schema.encode(Schema.parseJson(Schema.Struct({ format: LinuxPackageFormat })))({ format: options.format }))
  yield* fs.copyFile(join(root, "packages/release/resources/linux/launch.sh"), join(payload, "launch"))
  yield* fs.chmod(join(payload, "launch"), 0o755)
  const begin = yield* fs.readFileString(join(root, "packages/release/resources/linux/installation-begin.sh"))
  const end = yield* fs.readFileString(join(root, "packages/release/resources/linux/installation-end.sh"))
  const { format } = options
  const candidate = format === "pacman"
    ? yield* buildPacmanPackage(options, stage, payload, begin, end)
    : yield* buildElectronInstaller({ ...options, format }, stage, payload, begin, end)
  const filename = linuxDesktopInstaller(options.arch === "arm64" ? "linux-arm64-gnu" : "linux-x64-gnu", options.format, options.version, options.revision)
  yield* validateLinuxDesktopInstaller({ ...options, file: candidate })
  yield* fs.makeDirectory(options.output, { recursive: true })
  const output = resolve(options.output, filename)
  yield* fs.copyFile(candidate, output)
  const artifact = yield* Schema.decodeUnknown(ReleaseArtifactSchema)({
    id: `desktop-linux-${options.arch}-gnu-${options.format}`,
    kind: "desktop", host: `linux-${options.arch}-gnu`,
    filename: basename(output), bytes: Number((yield* fs.stat(output)).size),
    sha256: yield* sha256File(output),
  })
  yield* fs.writeFileString(join(options.output, `${artifact.id}.artifact.json`),
    yield* Schema.encode(Schema.parseJson(ReleaseArtifactSchema))(artifact), { flag: "wx", mode: 0o600 })
  return { output, artifact }
})).pipe(Effect.mapError(error => error instanceof DesktopBuildFailed ? error : new DesktopBuildFailed({ message: String(error) })))
