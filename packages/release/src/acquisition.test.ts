import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import * as NodePath from "@effect/platform-node/NodePath"
import { createHash } from "node:crypto"
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Layer, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  acquireRelease,
  installArtifact,
  validateReleaseManifestBytes,
} from "./acquisition"
import { NodeArchiveExtractor } from "./archive"
import { ReleaseArtifactSchema } from "./contracts"

const version = "1.2.3"
const tag = `@magnitudedev/cli@${version}`
const sourceCommit = "a".repeat(40)
const archive = new TextEncoder().encode("archive")
const sha256 = (bytes: Uint8Array) =>
  createHash("sha256").update(bytes).digest("hex")

const artifact = Schema.decodeUnknownSync(ReleaseArtifactSchema)({
  id: "cli-linux-x64-gnu",
  kind: "cli",
  host: "linux-x64-gnu",
  filename: "magnitude-cli-linux-x64-gnu.tar.gz",
  bytes: archive.byteLength,
  sha256: sha256(archive),
})

const manifestBytes = (
  overrides: Readonly<Record<string, unknown>> = {},
): Uint8Array =>
  new TextEncoder().encode(JSON.stringify({
    schemaVersion: 1,
    version,
    tag,
    sourceCommit,
    artifacts: [Schema.encodeSync(ReleaseArtifactSchema)(artifact)],
    ...overrides,
  }))

const AcquisitionLayer = Layer.mergeAll(
  NodeFileSystem.layer,
  NodePath.layer,
  FetchHttpClient.layer,
)
const InstallationLayer = Layer.mergeAll(
  AcquisitionLayer,
  NodeArchiveExtractor,
)

describe("unsigned release acquisition", () => {
  it("decodes a structurally valid manifest and retains its digest", async () => {
    const bytes = manifestBytes()
    const release = await Effect.runPromise(validateReleaseManifestBytes(bytes))

    expect(release.manifest.version).toBe(version)
    expect(release.manifest.tag).toBe(tag)
    expect(release.manifestSha256).toBe(sha256(bytes))
  })

  it("rejects a malformed manifest", async () => {
    const error = await Effect.runPromise(
      validateReleaseManifestBytes(new TextEncoder().encode("{}")).pipe(
        Effect.flip,
      ),
    )

    expect(error.stage).toBe("validate")
    expect(error.message).toBe("release manifest is malformed")
  })

  it("rejects a valid manifest for a different requested release", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-release-test-"))
    const wrongVersion = "9.9.9"
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: () =>
        new Response(new TextDecoder().decode(manifestBytes({
          version: wrongVersion,
          tag: `@magnitudedev/cli@${wrongVersion}`,
        }))),
    })
    try {
      const error = await Effect.runPromise(
        acquireRelease(
          `http://127.0.0.1:${server.port}`,
          version,
          join(root, "manifest"),
        ).pipe(
          Effect.provide(AcquisitionLayer),
          Effect.flip,
        ),
      )
      expect(error.stage).toBe("validate")
      expect(error.message).toBe("release identity differs from requested version")
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("reuses a valid cached manifest without another download", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-release-test-"))
    let requests = 0
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: () => {
        requests += 1
        return new Response(new TextDecoder().decode(manifestBytes()))
      },
    })
    const acquire = () =>
      Effect.runPromise(
        acquireRelease(
          `http://127.0.0.1:${server.port}`,
          version,
          join(root, "manifest"),
        ).pipe(Effect.provide(AcquisitionLayer)),
      )
    try {
      await acquire()
      await acquire()
      expect(requests).toBe(1)
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it.each([
    ["digest", "corrupt", "release artifact digest or size differs from manifest"],
    ["size", "short", "release artifact size differs from its manifest"],
  ])("rejects artifact bytes that differ from the manifest %s", async (_field, body, message) => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-release-test-"))
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: () => new Response(body),
    })
    try {
      const error = await Effect.runPromise(
        installArtifact(
          `http://127.0.0.1:${server.port}`,
          version,
          artifact,
          join(root, "installation"),
        ).pipe(
          Effect.provide(InstallationLayer),
          Effect.flip,
        ),
      )
      expect(error.stage).toBe("download")
      expect(error.message).toBe(message)
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })

  it("publishes an extracted installation without scoped staging cleanup failure", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-release-test-"))
    const payload = join(root, "payload")
    const archivePath = join(root, "cli.tar.gz")
    const destination = join(root, "installations", "digest")
    await mkdir(join(payload, "bin"), { recursive: true })
    const executable = join(payload, "bin", "magnitude-cli")
    await writeFile(executable, "#!/bin/sh\nexit 0\n")
    await chmod(executable, 0o755)
    const tar = Bun.spawn([
      "tar",
      "-czf",
      archivePath,
      "-C",
      payload,
      "bin/magnitude-cli",
    ], {
      stdout: "pipe",
      stderr: "pipe",
      env: { ...process.env, COPYFILE_DISABLE: "1" },
    })
    const [tarCode, tarError] = await Promise.all([
      tar.exited,
      new Response(tar.stderr).text(),
    ])
    expect(tarCode, tarError).toBe(0)
    const archiveBytes = new Uint8Array(await readFile(archivePath))
    const installable = Schema.decodeUnknownSync(ReleaseArtifactSchema)({
      id: "cli-linux-x64-gnu",
      kind: "cli",
      host: "linux-x64-gnu",
      filename: "magnitude-cli-linux-x64-gnu.tar.gz",
      bytes: archiveBytes.byteLength,
      sha256: sha256(archiveBytes),
    })
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: () => new Response(archiveBytes),
    })
    try {
      await Effect.runPromise(
        installArtifact(
          `http://127.0.0.1:${server.port}`,
          version,
          installable,
          destination,
        ).pipe(Effect.provide(InstallationLayer)),
      )
      expect(await readFile(join(destination, "bin", "magnitude-cli"), "utf8"))
        .toBe("#!/bin/sh\nexit 0\n")
      expect(
        (await readdir(join(root, "installations")))
          .filter((entry) => entry.startsWith(".release-")),
      ).toEqual([])
    } finally {
      server.stop(true)
      await rm(root, { recursive: true, force: true })
    }
  })
})
