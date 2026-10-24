import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { HardwareSummary } from "./machine-summary"
import { hardware, hardwareScenarios, identity } from "../../test/hardware/fixtures"

const mac = hardware("Apple M4 Max", 64, 16, [{ name: "Apple M4 Max", memory: 64, shared: true }], 16)

describe("detected machine expansion", () => {
  it("identifies the user's machine and keeps specs out of the closed summary", () => {
    const html = renderToStaticMarkup(<HardwareSummary value={mac} identity={identity("Apple Inc.", "Mac16,5", "Portable")} />)
    expect(html).toContain("Your machine:")
    expect(html).toContain("16-inch MacBook Pro")
    expect(html).toContain('aria-expanded="false"')
    expect(html).not.toContain("Memory bandwidth")
    expect(html).not.toContain("<dl")
  })
  it.each(hardwareScenarios)("retains the real component facts for $label", scenario => {
    const html = renderToStaticMarkup(<HardwareSummary value={scenario.value} identity={scenario.identity} expanded />)
    expect(html).toContain('aria-expanded="true"')
    expect(html).toContain("Your machine:")
    expect(html).toContain("Memory")
    expect(html).not.toContain('role="dialog"')
    expect(html).not.toContain("Machine details")
    expect(html).not.toContain("<figure")
    if (scenario.value.accelerators.length === 0) {
      expect(html).toContain("CPU inference")
      expect(html).not.toContain(">GPU</dt>")
    }
  })
  it("retains bandwidth and both CPU and GPU cores without duplicating the integrated chip", () => {
    const html = renderToStaticMarkup(<HardwareSummary value={mac} identity={identity("Apple Inc.", "Mac16,5", "Portable")} expanded />)
    expect(html).toContain("Memory bandwidth")
    expect(html).toContain("546 GB/s")
    expect(html).toContain("CPU cores")
    expect(html).toContain("GPU cores")
    expect(html).not.toContain(">GPU</dt>")
    expect(html).toContain("unified memory")
  })
  it("keeps each GPU's VRAM separate from system RAM", () => {
    const value = hardware("AMD Ryzen 9 9950X", 128, 32, [{ name: "NVIDIA GeForce RTX 5090", memory: 32 }, { name: "NVIDIA GeForce RTX 5090", memory: 32 }], 16)
    const html = renderToStaticMarkup(<HardwareSummary value={value} identity={identity("Custom", "Workstation", "Desktop")} expanded />)
    expect(html).toContain("32 GiB VRAM each")
    expect(html).toContain("GPU 1")
    expect(html).toContain("GPU 2")
    expect(html).not.toContain("64 GiB VRAM")
    expect(html).toContain("128 GiB")
  })
  it("does not confuse two GPUs with the same name but different VRAM", () => {
    const value = hardware("AMD Ryzen 9 9950X", 128, 32, [{ name: "Example GPU", memory: 24 }, { name: "Example GPU", memory: 32 }], 16)
    const html = renderToStaticMarkup(<HardwareSummary value={value} identity={null} expanded />)
    expect(html).toContain("24 GiB / 32 GiB VRAM")
    const gpu1 = html.indexOf(">GPU 1</dt>")
    const gpu2 = html.indexOf(">GPU 2</dt>")
    expect(html.slice(gpu1, gpu2)).toContain("24 GiB")
    expect(html.slice(gpu2)).toContain("32 GiB")
    const activeSecond = { ...value, activeDeviceId: value.accelerators[1]!.executors[0]!.deviceId }
    const reordered = renderToStaticMarkup(<HardwareSummary value={activeSecond} identity={null} expanded />)
    expect(reordered).toContain("32 GiB / 24 GiB VRAM")
    const firstGpu = reordered.slice(reordered.indexOf(">GPU 1</dt>"), reordered.indexOf(">GPU 2</dt>"))
    expect(firstGpu).toContain("32 GiB")
    expect(firstGpu).toContain("In use")
    const cpuDevice = hardware("AMD Ryzen 9 9950X", 128, 32).activeDeviceId
    const onCpu = renderToStaticMarkup(<HardwareSummary value={{ ...value, activeDeviceId: cpuDevice }} identity={null} expanded />)
    expect(onCpu).toContain("CPU inference")
    expect(onCpu).toContain("Available")
    expect(onCpu).not.toContain("In use")
  })

})
