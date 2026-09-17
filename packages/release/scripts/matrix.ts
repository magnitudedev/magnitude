import { appendFile } from "node:fs/promises"
import { backendPacks, releaseHosts } from "../src/targets"

const selectedPacks = process.argv.includes("--windows-backends-only")
  ? backendPacks.filter(pack => pack.host === "windows-x64-msvc")
  : backendPacks

const matrices = {
  hosts: {
    include: releaseHosts.map((host) => ({
      id: host.id,
      runner: host.runner,
      rustTarget: host.rustTarget,
    })),
  },
  backends: {
    include: selectedPacks.map((pack) => ({
      id: pack.id,
      host: pack.host,
      backend: pack.backend,
      runner: pack.runner,
      toolkit: "cuda" in pack ? pack.cuda.toolkitVersion : "",
    })),
  },
  appleHosts: {
    include: releaseHosts.filter((host) => host.id.startsWith("darwin-")).map((host) => ({ id: host.id, runner: host.runner })),
  },
  linuxHosts: {
    include: releaseHosts
      .filter((host) => host.id.startsWith("linux-"))
      .map((host) => ({
        id: host.id,
        runner: host.id === "linux-arm64-gnu"
          ? "blacksmith-8vcpu-ubuntu-2204-arm"
          : "blacksmith-8vcpu-ubuntu-2204",
      })),
  },
}

const output = process.env.GITHUB_OUTPUT
if (output) {
  await appendFile(output, `hosts=${JSON.stringify(matrices.hosts)}\n`)
  await appendFile(output, `backends=${JSON.stringify(matrices.backends)}\n`)
  await appendFile(output, `appleHosts=${JSON.stringify(matrices.appleHosts)}\n`)
  await appendFile(output, `linuxHosts=${JSON.stringify(matrices.linuxHosts)}\n`)
} else {
  console.log(JSON.stringify(matrices, null, 2))
}
