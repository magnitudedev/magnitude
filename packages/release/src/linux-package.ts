import { Schema } from "effect"

/**
 * Newest glibc and libstdc++ symbol versions any shipped Linux binary may require, so RHEL 9
 * (glibc 2.34, GLIBCXX_3.4.29) runs them. Every package declares the glibc version as its floor.
 */
export const LINUX_GLIBC_MINIMUM = "2.34"
export const LINUX_GLIBCXX_MAXIMUM = "3.4.29"

export const LinuxPackageFormat = Schema.Literal("deb", "rpm", "pacman")
export type LinuxPackageFormat = typeof LinuxPackageFormat.Type
export type LinuxPackageArch = "arm64" | "x64"

/** Arch Linux and its derivatives publish x86_64 only, so pacman packages exist only for that host. */
export const linuxPackageFormats = (arch: LinuxPackageArch): readonly LinuxPackageFormat[] =>
  arch === "x64" ? ["deb", "rpm", "pacman"] : ["deb", "rpm"]

export const linuxPackageArchitecture = (format: LinuxPackageFormat, arch: LinuxPackageArch): string =>
  format === "deb" ? (arch === "arm64" ? "arm64" : "amd64") : (arch === "arm64" ? "aarch64" : "x86_64")

/**
 * Package-manager version for a SemVer release. Dpkg and rpm order `~` before the release it
 * precedes. Pacman treats `~` as an ordinary separator, so 1.0.0~beta.1 would sort after 1.0.0;
 * it orders a letter suffix that directly follows the release number before that release instead.
 */
export const linuxPackageVersion = (format: LinuxPackageFormat, version: string): string =>
  version.replace("-", format === "pacman" ? "" : "~")

export const linuxPackageExtension = (format: LinuxPackageFormat): string =>
  format === "pacman" ? ".pkg.tar.zst" : `.${format}`

/** Name, version and architecture from a pacman package's .PKGINFO, tab-separated like the dpkg and rpm queries. */
export const pacmanPackageIdentity = (info: string) => {
  const fields = new Map<string, string>()
  for (const line of info.split("\n")) {
    const field = /^(pkgname|pkgver|arch) = (.+)$/.exec(line)
    if (field) fields.set(field[1]!, fields.has(field[1]!) ? "" : field[2]!)
  }
  return `${fields.get("pkgname") ?? ""}\t${fields.get("pkgver") ?? ""}\t${fields.get("arch") ?? ""}`
}
