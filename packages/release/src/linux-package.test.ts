import { describe, expect, it } from "vitest"
import { linuxPackageArchitecture, linuxPackageExtension, linuxPackageFormats, linuxPackageVersion, pacmanPackageIdentity } from "./linux-package"

describe("Linux package formats", () => {
  it("publishes pacman packages for x86-64 only", () => {
    expect(linuxPackageFormats("x64")).toEqual(["deb", "rpm", "pacman"])
    expect(linuxPackageFormats("arm64")).toEqual(["deb", "rpm"])
  })

  it.each([
    ["deb", "x64", "amd64", ".deb"], ["deb", "arm64", "arm64", ".deb"],
    ["rpm", "x64", "x86_64", ".rpm"], ["rpm", "arm64", "aarch64", ".rpm"],
    ["pacman", "x64", "x86_64", ".pkg.tar.zst"],
  ] as const)("names %s %s packages", (format, arch, architecture, extension) => {
    expect(linuxPackageArchitecture(format, arch)).toBe(architecture)
    expect(linuxPackageExtension(format)).toBe(extension)
  })

  // Pacman's vercmp, observed on Arch Linux: 0.2.7~beta.1 > 0.2.7 but 0.2.7beta.1 < 0.2.7.
  it.each([
    ["deb", "0.2.7-beta.1", "0.2.7~beta.1"],
    ["rpm", "0.2.7-alpha.2", "0.2.7~alpha.2"],
    ["pacman", "0.2.7-beta.1", "0.2.7beta.1"],
    ["pacman", "0.2.7", "0.2.7"],
  ] as const)("orders %s prerelease %s as %s", (format, version, expected) => {
    expect(linuxPackageVersion(format, version)).toBe(expected)
  })

  it("reads pacman identity only from unambiguous metadata", () => {
    expect(pacmanPackageIdentity("pkgname = magnitude-desktop\npkgver = 1.0.0-3\narch = x86_64\ndepend = glibc\n")).toBe("magnitude-desktop\t1.0.0-3\tx86_64")
    expect(pacmanPackageIdentity("pkgname = magnitude-desktop\npkgname = other\npkgver = 1.0.0-3\narch = x86_64\n")).toBe("\t1.0.0-3\tx86_64")
    expect(pacmanPackageIdentity("pkgname=magnitude-desktop\n")).toBe("\t\t")
  })
})
