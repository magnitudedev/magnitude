import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { afterEach, describe, expect, it } from "vitest"
import { downloadAcn, immutableBinaryPath } from "./binary"

describe.runIf(process.platform !== "win32")("immutable ACN binary cache", () => {
  let server: ReturnType<typeof Bun.serve> | undefined
  const previousBase = process.env.MAGNITUDE_RELEASE_BASE_URL

  afterEach(() => {
    server?.stop(true)
    server = undefined
    if (previousBase === undefined) delete process.env.MAGNITUDE_RELEASE_BASE_URL
    else process.env.MAGNITUDE_RELEASE_BASE_URL = previousBase
  })

  it("atomically converges concurrent downloads on a version/platform path", async () => {
    const directory = await mkdtemp(join(tmpdir(), "magnitude-acn-cache-"))
    const payload = join(directory, "payload")
    const executable = join(payload, "magnitude-acn")
    const archive = join(directory, "magnitude-acn.tar.gz")
    await mkdir(payload, { recursive: true })
    await writeFile(executable, "#!/bin/sh\nprintf '1.2.3\\n'\n")
    await chmod(executable, 0o755)
    const tar = Bun.spawn(["tar", "-czf", archive, "-C", payload, "magnitude-acn"])
    expect(await tar.exited).toBe(0)
    server = Bun.serve({ port: 0, fetch: () => new Response(Bun.file(archive)) })
    process.env.MAGNITUDE_RELEASE_BASE_URL = `http://127.0.0.1:${server.port}`

    try {
      const run = downloadAcn("1.2.3", directory).pipe(
        Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      )
      const paths = await Effect.runPromise(
        Effect.all([run, run], { concurrency: "unbounded" }),
      )
      expect(paths[0]).toBe(immutableBinaryPath(directory, "1.2.3"))
      expect(paths[1]).toBe(paths[0])

      server.stop(true)
      server = undefined
      expect(await Effect.runPromise(run)).toBe(paths[0])
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })
})
