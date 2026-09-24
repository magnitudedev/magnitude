import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync, existsSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { spawnSync } from "node:child_process"
import { describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../src/hosted-update/release"
import { renderUnixInstallationScript } from "./installation-scripts"

describe.skipIf(process.platform !== "linux")("Linux installation shell verification", () => {
  it.each(["valid", "signature", "digest", "architecture", "channel"])("admits the package manager only for %s input", async scenario => {
    const root = mkdtempSync(join(tmpdir(), "magnitude shell ' "))
    try {
      const bin = join(root, "bin")
      mkdirSync(bin)
      const executable = (name: string, contents: string) => writeFileSync(join(bin, name), contents, { mode: 0o700 })
      // Only network and privileged package mutation are replaced; parsing and OpenSSL run natively.
      executable("curl", '#!/bin/sh\nwhile [ "$#" -gt 0 ]; do\n case "$1" in --output) output=$2; shift 2;; *) url=$1; shift;; esac\ndone\ncase "$url" in *.json) cp "$TEST_OFFER" "$output";; *) cp "$TEST_PACKAGE" "$output";; esac\n')
      executable("apt-get", '#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_APT_RECEIPT"\n')
      executable("sudo", '#!/bin/sh\nexec "$@"\n')
      const keys = generateKeyPairSync("ed25519")
      const bytes = Buffer.from("verified package fixture")
      const packagePath = join(root, "package.deb"), offerPath = join(root, "offer.json"), receipt = join(root, "receipt")
      writeFileSync(packagePath, scenario === "digest" ? Buffer.from("modified package fixture") : bytes)
      const arch = process.arch === "arm64" ? "arm64" : "x64"
      const release = await Effect.runPromise(signUpdateRelease({ version: scenario === "channel" ? "0.1.6-beta.1" : "0.1.6",
        bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") },
      { os: "linux", arch: scenario === "architecture" ? (arch === "arm64" ? "x64" : "arm64") : arch, package: "deb" }, keys.privateKey))
      writeFileSync(offerPath, JSON.stringify({ release: scenario === "signature" ? { ...release, signature: "A".repeat(86) + "==" } : release,
        download: "https://github.com/magnitudedev/magnitude/releases/download/test/magnitude.deb" }))
      const script = await Effect.runPromise(renderUnixInstallationScript({ origin: "https://magnitude.dev", appleTeam: "ABCDEFGHIJ",
        publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString() }).pipe(Effect.provide(NodeContext.layer)))
      const result = spawnSync("/bin/sh", ["-s"], { input: script, encoding: "utf8", timeout: 15000,
        env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, TEST_OFFER: offerPath, TEST_PACKAGE: packagePath, TEST_APT_RECEIPT: receipt } })
      if (scenario === "valid") {
        expect(result.status, result.stderr).toBe(0)
        expect(readFileSync(receipt, "utf8")).toMatch(/^install\n-y\n.*magnitude\.deb\n$/)
      } else {
        expect(result.status).not.toBe(0)
        expect(existsSync(receipt)).toBe(false)
      }
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})
