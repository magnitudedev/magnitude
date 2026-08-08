import { chmod, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { spawnSync } from "node:child_process"
import { tmpdir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { afterEach, describe, expect, it } from "vitest"

const roots: string[] = []
const testDirectory = dirname(fileURLToPath(import.meta.url))

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

const runLauncher = async (
  environment: Readonly<Record<string, string>>,
  pnpmLayout = false,
): Promise<Record<string, string | undefined>> => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-launcher-"))
  roots.push(root)
  const nodeModules = join(root, "node_modules")
  const packageRoot = join(nodeModules, "@magnitudedev", "cli")
  const binDirectory = join(packageRoot, "bin")
  const libDirectory = join(packageRoot, "lib")
  const outputPath = join(root, "environment.json")
  const binaryPath = join(root, "fake-magnitude")
  await mkdir(binDirectory, { recursive: true })
  await mkdir(libDirectory, { recursive: true })
  await copyFile(
    resolve(testDirectory, "../../cli/bin/magnitude.js"),
    join(binDirectory, "magnitude.js"),
  )
  await writeFile(
    join(packageRoot, "package.json"),
    JSON.stringify({ name: "@magnitudedev/cli", version: "1.0.0" }),
  )
  await writeFile(
    join(libDirectory, "download.js"),
    "exports.ensureBinary = async () => process.env.TEST_MAGNITUDE_BINARY;\n",
  )
  await writeFile(binaryPath, `#!/usr/bin/env node
const { writeFileSync } = require("node:fs")
writeFileSync(process.env.TEST_MAGNITUDE_OUTPUT, JSON.stringify({
  npm: process.env.MAGNITUDE_MANAGED_BY_NPM,
  bun: process.env.MAGNITUDE_MANAGED_BY_BUN,
  pnpm: process.env.MAGNITUDE_MANAGED_BY_PNPM,
  packageRoot: process.env.MAGNITUDE_MANAGED_PACKAGE_ROOT,
}))
`)
  await chmod(binaryPath, 0o755)
  if (pnpmLayout) await writeFile(join(nodeModules, ".modules.yaml"), "")

  const result = spawnSync(
    process.execPath,
    [join(binDirectory, "magnitude.js")],
    {
      encoding: "utf8",
      env: {
        ...process.env,
        TEST_MAGNITUDE_BINARY: binaryPath,
        TEST_MAGNITUDE_OUTPUT: outputPath,
        ...environment,
      },
    },
  )
  expect(result.status, result.stderr).toBe(0)
  return JSON.parse(await readFile(outputPath, "utf8"))
}

describe("npm launcher installation context", () => {
  it("passes npm provenance to the native CLI", async () => {
    const result = await runLauncher({
      npm_config_user_agent: "npm/11.0.0 node/v24",
      npm_execpath: "/usr/bin/npm",
    })
    expect(result).toMatchObject({ npm: "1", packageRoot: expect.any(String) })
    expect(result.bun).toBeUndefined()
    expect(result.pnpm).toBeUndefined()
  })

  it("passes Bun provenance to the native CLI", async () => {
    const result = await runLauncher({ npm_config_user_agent: "bun/1.3.0 npm/?" })
    expect(result.bun).toBe("1")
    expect(result.npm).toBeUndefined()
  })

  it("prefers a pnpm-owned package layout", async () => {
    const result = await runLauncher(
      {
        npm_config_user_agent: "npm/11.0.0 node/v24",
        npm_execpath: "/usr/bin/npm",
      },
      true,
    )
    expect(result.pnpm).toBe("1")
    expect(result.npm).toBeUndefined()
  })
})
