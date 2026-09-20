import { Effect, Schema } from "effect"
import { join } from "node:path"
import { Candidate } from "../candidate"
import { AssertionFailure, Target } from "../domain"
import { InstalledApplication } from "../installer"
import { checkedCommand } from "../process"

const fail = (message: string) => new AssertionFailure({ message })
/** Read the native package database; never install the proposed replacement to satisfy an update. */
export const observeUpdatedInstallation = (previous: InstalledApplication, candidate: Candidate, environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  if (!Schema.equivalence(Target)(previous.candidate.target, candidate.target)) return yield* fail("Updated installation target changed")
  const run = (name: string, args: readonly string[]) => checkedCommand(name, args, { env: { ...environment }, inheritEnv: false, timeoutMs: 30_000 })
  const format = candidate.target.packageFormat
  let version = candidate.version
  if (format === "deb") {
    const expected = (yield* run("dpkg-deb", ["-f", candidate.path, "Version"])).stdout.trim()
    version = (yield* run("dpkg-query", ["-W", "-f=${Version}", "magnitude-desktop"])).stdout.trim()
    if (version !== expected) return yield* fail("Native package database still contains a different DEB version")
  } else if (format === "rpm") {
    const expected = (yield* run("rpm", ["-qp", "--qf", "%{VERSION}-%{RELEASE}", candidate.path])).stdout.trim()
    version = (yield* run("rpm", ["-q", "--qf", "%{VERSION}-%{RELEASE}", "magnitude-desktop"])).stdout.trim()
    if (version !== expected) return yield* fail("Native package database still contains a different RPM version")
  } else if (format === "dmg") {
    version = (yield* run("/usr/bin/plutil", ["-extract", "CFBundleShortVersionString", "raw", "-o", "-", join(previous.root, "Contents/Info.plist")])).stdout.trim()
    if (version !== candidate.version) return yield* fail("Updated bundle version differs from its candidate")
  }
  const cli = yield* run(previous.cli, ["--version"])
  if (cli.stdout.trim() !== candidate.version) return yield* fail("Updated bundled CLI version differs from its candidate")
  return InstalledApplication.make({ ...previous, candidate, packageVersion: version })
})
