import { appendFile } from "node:fs/promises"
import { releaseHosts } from "../src/targets"

const matrices = {
  hosts: {
    include: releaseHosts.map((host) => ({
      id: host.id,
      runner: host.runner,
      rustTarget: host.rustTarget,
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
  await appendFile(output, `appleHosts=${JSON.stringify(matrices.appleHosts)}\n`)
  await appendFile(output, `linuxHosts=${JSON.stringify(matrices.linuxHosts)}\n`)
} else {
  console.log(JSON.stringify(matrices, null, 2))
}
