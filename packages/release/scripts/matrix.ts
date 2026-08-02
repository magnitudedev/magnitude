import { appendFile } from "node:fs/promises"
import { backendPacks, releaseHosts } from "../src/targets"

const backendEntry = (pack: typeof backendPacks[number]) => ({
  id: pack.id,
  host: pack.host,
  backend: pack.backend,
  runner: pack.runner,
  toolkit: "cuda" in pack ? pack.cuda.toolkitVersion : "",
})

const isCuda11 = (pack: typeof backendPacks[number]): boolean =>
  pack.backend === "cuda" && pack.cuda.toolkitVersion === "11.8"

const matrices = {
  hosts: {
    include: releaseHosts.map((host) => ({
      id: host.id,
      runner: host.runner,
      rustTarget: host.rustTarget,
    })),
  },
  backends: {
    include: backendPacks.filter((pack) => !isCuda11(pack)).map(backendEntry),
  },
  cuda11Backends: {
    include: backendPacks.filter(isCuda11).map(backendEntry),
  },
}

const output = process.env.GITHUB_OUTPUT
if (output) {
  await appendFile(output, `hosts=${JSON.stringify(matrices.hosts)}\n`)
  await appendFile(output, `backends=${JSON.stringify(matrices.backends)}\n`)
  await appendFile(
    output,
    `cuda11_backends=${JSON.stringify(matrices.cuda11Backends)}\n`,
  )
} else {
  console.log(JSON.stringify(matrices, null, 2))
}
