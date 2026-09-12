import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Schema } from "effect"
import { basename, dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { DesktopBuildFailed } from "./desktop"
import { LINUX_DESKTOP_PACKAGE_NAME } from "../../src/executables"
import { ReleaseArtifactSchema } from "../../src/contracts"
import { sha256File } from "../../src/macos-app"
import { linuxDesktopInstaller } from "../../src/targets"

const root = resolve(import.meta.dir, "../../../..")
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
const DebianOptions = Schema.Struct({ options: Schema.Struct({ ...PackageFields, depends: Schema.Array(Schema.String), scripts: Schema.Record({ key: Schema.String, value: Schema.String }) }) })
const RpmOptions = Schema.Struct({ options: Schema.Struct({ ...PackageFields, requires: Schema.Array(Schema.String), license: Schema.String, specTemplate: Schema.String }) })

export const validateLinuxDesktopInstaller = (options: {
  readonly file: string
  readonly format: "deb" | "rpm"
  readonly arch: "arm64" | "x64"
  readonly version: string
  readonly revision: number
}) => Effect.gen(function* () {
  const arch = options.format === "deb" ? (options.arch === "arm64" ? "arm64" : "amd64") : (options.arch === "arm64" ? "aarch64" : "x86_64")
  const identity = yield* (options.format === "deb"
    ? Command.make("dpkg-deb", "--show", "--showformat=${Package}\t${Version}\t${Architecture}", options.file)
    : Command.make("rpm", "-qp", "--qf", "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}", options.file)).pipe(Command.string)
  const expected = `${LINUX_DESKTOP_PACKAGE_NAME}\t${options.version.replace("-", "~")}-${options.revision}\t${arch}`
  if (identity.trim() !== expected) return yield* new DesktopBuildFailed({ message: "Linux desktop package identity does not match its release target" })
  if (options.format === "deb") {
    const listing = yield* Command.make("dpkg-deb", "--contents", options.file).pipe(Command.env({ LC_ALL: "C" }), Command.string)
    const sandbox = listing.split("\n").find(line => line.endsWith(` ./usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/chrome-sandbox`))
    if (sandbox === undefined || !/^-rwsr-xr-x\s+root\/root\s/.test(sandbox)) {
      return yield* new DesktopBuildFailed({ message: "Debian installer must contain a root-owned mode-04755 Chromium sandbox helper" })
    }
  } else {
    const listing = yield* Command.make("rpm", "-qp", "--qf", "[%{FILENAMES} %{FILEMODES:octal} %{FILEUSERNAME} %{FILEGROUPNAME}\n]", options.file).pipe(Command.string)
    if (!listing.split("\n").includes(`/usr/lib/${LINUX_DESKTOP_PACKAGE_NAME}/chrome-sandbox 104755 root root`)) {
      return yield* new DesktopBuildFailed({ message: "RPM installer must contain a root-owned mode-04755 Chromium sandbox helper" })
    }
  }
})

/** Package an already assembled, matched app/service. Publication requires native consumer acceptance. */
export const buildLinuxDesktopInstaller = (options: {
  readonly format: "deb" | "rpm"
  readonly version: string
  readonly app: string
  readonly output: string
  readonly arch: "arm64" | "x64"
  readonly revision: number
}) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const stage = yield* fs.makeTempDirectoryScoped({ prefix: `magnitude-desktop-${options.format}-` })
  const config = join(stage, "options.json")
  const payload = join(stage, "application")
  yield* fs.copy(resolve(options.app), payload)
  yield* fs.copyFile(join(root, "packages/release/resources/linux/launch.sh"), join(payload, "launch"))
  yield* fs.chmod(join(payload, "launch"), 0o755)
  const begin = yield* fs.readFileString(join(root, "packages/release/resources/linux/installation-begin.sh"))
  const end = yield* fs.readFileString(join(root, "packages/release/resources/linux/installation-end.sh"))
  const scripts: Record<string, string> = {}
  for (const [name, body] of Object.entries({
    preinst: `case "$1" in install|upgrade) ;; *) exit 0 ;; esac\n${begin}`,
    postinst: `case "$1" in configure) ;; *) exit 0 ;; esac\n${end}`,
    prerm: `case "$1" in remove) ;; *) exit 0 ;; esac\n${begin}`,
    postrm: `case "$1" in remove|purge) ;; *) exit 0 ;; esac\n[ -f /var/lib/magnitude-desktop/installation.lock ] || exit 0\n${end}`,
  })) {
    scripts[name] = join(stage, name)
    yield* fs.writeFileString(scripts[name]!, `#!/bin/sh\nset -eu\n${body}`)
    yield* fs.chmod(scripts[name]!, 0o755)
  }
  const specTemplate = join(stage, "desktop.spec.ejs")
  const spec = yield* fs.readFileString(join(root, "packages/release/resources/linux/desktop.spec.ejs"))
  yield* fs.writeFileString(specTemplate, spec.replaceAll("@MAGNITUDE_INSTALL_BEGIN@", begin).replaceAll("@MAGNITUDE_INSTALL_END@", end))
  const desktopTemplate = join(stage, "magnitude.desktop.ejs")
  yield* fs.writeFileString(desktopTemplate, `[Desktop Entry]\nName=Magnitude\nComment=Discover and run local models\nExec=${LINUX_DESKTOP_PACKAGE_NAME}\nIcon=${LINUX_DESKTOP_PACKAGE_NAME}\nType=Application\nTerminal=false\nStartupNotify=true\nStartupWMClass=${LINUX_DESKTOP_PACKAGE_NAME}\nCategories=Development;\n`)
  const metadata = {
    name: LINUX_DESKTOP_PACKAGE_NAME, productName: "Magnitude", genericName: "Local inference",
    description: "Discover and run local models",
    productDescription: "Magnitude provides curated model discovery, local inference, and configuration for external agent harnesses.",
    maintainer: "Magnitude <founders@magnitude.dev>", homepage: "https://magnitude.dev",
    bin: "launch", revision: String(options.revision),
    icon: { "256x256": join(root, "assets/brand/application-icon.png") },
    categories: ["Development"], desktopTemplate,
    version: options.version.replace("-", "~"),
  } as const
  yield* fs.writeFileString(config, options.format === "deb"
    ? yield* Schema.encode(Schema.parseJson(DebianOptions))({ options: { ...metadata, depends: ["libc6 (>= 2.35)", "libasound2t64 | libasound2", "util-linux"], scripts } })
    : yield* Schema.encode(Schema.parseJson(RpmOptions))({ options: { ...metadata, requires: ["glibc >= 2.35", "alsa-lib", "util-linux"], license: "Apache-2.0", specTemplate } }))
  const destination = join(stage, "packages")
  const tool = options.format === "deb" ? "electron-installer-debian" : "electron-installer-redhat"
  const cli = join(dirname(fileURLToPath(import.meta.resolve(tool))), "cli.js")
  const arch = options.format === "deb" ? (options.arch === "arm64" ? "arm64" : "amd64") : (options.arch === "arm64" ? "aarch64" : "x86_64")
  // The installer uses fs.chmod for the setuid sandbox. Bun 1.3.14 drops those
  // special mode bits on Linux; run this native packaging boundary under Node.
  const code = yield* Command.make("node", cli, "--src", payload, "--dest", destination,
    "--arch", arch, "--config", config).pipe(
    Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
  )
  if (code !== 0) return yield* new DesktopBuildFailed({ message: `${options.format} desktop packaging exited ${code}` })
  const packages = (yield* fs.readDirectory(destination)).filter(path => path.endsWith(`.${options.format}`))
  if (packages.length !== 1) return yield* new DesktopBuildFailed({ message: `${options.format} desktop packaging did not produce exactly one installer` })
  if (packages[0] !== linuxDesktopInstaller(options.arch === "arm64" ? "linux-arm64-gnu" : "linux-x64-gnu", options.format, options.version, options.revision)) {
    return yield* new DesktopBuildFailed({ message: "Linux desktop installer filename does not match its release target" })
  }
  const candidate = join(destination, packages[0]!)
  yield* validateLinuxDesktopInstaller({ ...options, file: candidate })
  yield* fs.makeDirectory(options.output, { recursive: true })
  const output = resolve(options.output, packages[0]!)
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
