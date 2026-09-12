import { expect, it } from "vitest"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { createRequire } from "node:module"
import { resolve } from "node:path"

it("uses Electron SQLite for bound values, passive contention, scoped release, and no-create opens", async () => {
  const root = await mkdtemp(`${tmpdir()}/magnitude-electron-sqlite-`)
  try {
    const build = await Bun.build({ entrypoints: [resolve(import.meta.dirname, "fixtures/node-sqlite.ts")], outdir: root, target: "node", format: "esm" })
    expect(build.success, String(build.logs)).toBe(true)
    const electron: string = createRequire(import.meta.url)("electron")
    const process = Bun.spawn([electron, `${root}/node-sqlite.js`], { env: { ...Bun.env, ELECTRON_RUN_AS_NODE: "1" }, stdout: "pipe", stderr: "pipe" })
    const [code, stdout, stderr] = await Promise.all([process.exited, new Response(process.stdout).text(), new Response(process.stderr).text()])
    expect(code, stderr).toBe(0)
    expect(stdout).toContain("Node SQLite acceptance passed")
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
