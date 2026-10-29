import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { renderServeReady, type ServerReach } from "./server-reach"

const reach = (overrides: Partial<ServerReach> = {}): ServerReach => ({
  endpoint: "http://127.0.0.1:10100", configPath: "/home/ada/.magnitude/config.json", service: false,
  network: Option.some({ enabled: true, bind: Option.none(), port: 10100, pending: false, container: false, interfaces: [
    { name: "eth0", address: "192.168.1.5", kind: "lan" },
    { name: "tailscale0", address: "100.64.0.3", kind: "tailscale" },
    { name: "docker0", address: "172.17.0.1", kind: "virtual" },
  ] }),
  ...overrides,
})

describe("serve ready output", () => {
  it("lists this computer and every address other devices can use", () => {
    const output = renderServeReady(reach())
    expect(output).toContain("On this computer:   http://127.0.0.1:10100")
    expect(output).toContain("  http://192.168.1.5:10100")
    expect(output).toContain("  http://100.64.0.3:10100  (Tailscale)")
    expect(output).not.toContain("172.17.0.1")
    expect(output).toContain("asks for the network access key")
    expect(output).toContain("Apps and agents:    http://192.168.1.5:10100/inference/v1")
    expect(output).toContain("Press Ctrl+C to stop.")
  })
  it("explains how to turn network access on with the profile's real config path, and the SSH tunnel", () => {
    const off = Option.some({ enabled: false, bind: Option.none(), port: 10100, pending: false, container: false, interfaces: [] })
    const server = renderServeReady(reach({ network: off, configPath: "/var/lib/magnitude/config.json", service: true }))
    expect(server).toContain("Only this computer can reach it.")
    expect(server).toContain("/var/lib/magnitude/config.json")
    expect(server).toContain("ssh -N -L 10100:127.0.0.1:10100")
    expect(renderServeReady(reach({ network: off }))).toContain("/home/ada/.magnitude/config.json")
  })
  it("tunnels to the port the service actually listens on", () => {
    const off = Option.some({ enabled: false, bind: Option.none(), port: 10100, pending: false, container: false, interfaces: [] })
    const output = renderServeReady(reach({ endpoint: "http://127.0.0.1:11188", network: off }))
    expect(output).toContain("ssh -N -L 11188:127.0.0.1:11188 <you>@<this machine>")
    expect(output).not.toContain("10100")
  })
  it("tells the service how to stop instead of Ctrl+C", () => {
    const linux = renderServeReady(reach({ service: true }), "linux")
    expect(linux).not.toContain("Ctrl+C")
    expect(linux).toContain("Stop it with `sudo systemctl stop magnitude`; `magnitude server remove` removes the server.")
    expect(renderServeReady(reach({ service: true }), "darwin")).toContain("Stop it with `magnitude server remove`.")
  })
  it("never prints the network access key", () => {
    const settings = { enabled: true, bind: Option.none(), port: 10100, pending: false, container: false, interfaces: [], apiKey: Option.some("secret-key"), requireApiKey: true }
    expect(renderServeReady(reach({ network: Option.some(settings) }))).not.toContain("secret-key")
  })
  it("describes a container's reach from the host and other containers, never its own addresses", () => {
    const settings = { enabled: true, bind: Option.none(), port: 10100, pending: false, container: true, interfaces: [{ name: "eth0", address: "172.17.0.2", kind: "lan" as const }] }
    const output = renderServeReady(reach({ network: Option.some(settings), configPath: "/data/.magnitude/config.json" }))
    expect(output).toContain("On the host:        http://localhost:10100, or the host port you published")
    expect(output).toContain("http://<this container's name>:10100")
    expect(output).toContain("docker exec <container> cat /data/.magnitude/config.json")
    expect(output).toContain("Stop it with `docker stop <container>`.")
    expect(output).not.toContain("172.17.0.2")
    expect(output).not.toContain("Ctrl+C")
    const off = renderServeReady(reach({ network: Option.some({ ...settings, enabled: false }), configPath: "/data/.magnitude/config.json" }))
    expect(off).toContain("Only this container can reach it")
    expect(off).not.toContain("ssh -N")
  })
  it("notes a network access change that waits for a restart", () => {
    expect(renderServeReady(reach({ network: Option.some({ enabled: false, bind: Option.none(), port: 10100, pending: true, container: false, interfaces: [] }) })))
      .toContain("applies when Magnitude restarts")
  })
})
