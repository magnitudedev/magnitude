import { expect, test } from "vitest"
import { Effect } from "effect"
import { linuxIdentity, nvidiaDevices, windowsIdentity } from "../src/host-inspector"

test("Linux identity preserves distro versions and DGX OTA identity without executing shell values", async () => {
  for (const [id, os, version] of [["ubuntu", "ubuntu", "24.04"], ["debian", "debian", "13"], ["fedora", "fedora", "44"], ["rhel", "redhat", "10.0"]]) {
    expect(await Effect.runPromise(linuxIdentity(`ID=${id}\nVERSION_ID="${version}"`))).toEqual({ os, version })
  }
  expect(await Effect.runPromise(linuxIdentity('ID=ubuntu\nVERSION_ID="24.04"', 'DGX_SWBUILD_VERSION="7.2.3"\nDGX_OTA_VERSION="7.5.0"'))).toEqual({ os: "dgx-os", version: "7.5.0" })
  for (const wire of ['ID=alpine\nVERSION_ID="3.20"', 'ID=ubuntu\nVERSION_ID="$(id)"', 'ID=ubuntu']) {
    expect((await Effect.runPromise(linuxIdentity(wire).pipe(Effect.either)))._tag).toBe("Left")
  }
})

test("Windows client identity distinguishes client versions and rejects Server despite matching build numbers", async () => {
  const report = { productType: 1, version: "10.0.26100", build: "26100", architecture: 9, cpuVendor: "GenuineIntel", cpuName: "Xeon", machineModel: "Virtual Machine", memoryBytes: 16 * 1024 ** 3 }
  expect(await Effect.runPromise(windowsIdentity(report))).toMatchObject({ os: "windows", version: "11", arch: "x64" })
  expect(await Effect.runPromise(windowsIdentity({ ...report, build: "19045", architecture: 12 }))).toMatchObject({ version: "10", arch: "arm64" })
  for (const invalid of [{ ...report, productType: 3 }, { ...report, productType: 2 }, { ...report, architecture: 0 }, { ...report, version: "6.3" }]) {
    expect((await Effect.runPromise(windowsIdentity(invalid).pipe(Effect.either)))._tag).toBe("Left")
  }
})

test("NVIDIA inventory records native identity and handles unavailable unified-memory totals honestly", async () => {
  expect(await Effect.runPromise(nvidiaDevices("GPU-one, NVIDIA A10, 580.1, 23028\nGPU-two, NVIDIA GB10, 580.1, [N/A]"))).toEqual([
    { uuid: "GPU-one", name: "NVIDIA A10", driver: "580.1", backend: "cuda", memoryBytes: 23028 * 1024 ** 2 },
    { uuid: "GPU-two", name: "NVIDIA GB10", driver: "580.1", backend: "cuda", memoryBytes: null },
  ])
  for (const line of ["bad output", "GPU-one, NVIDIA A10, 580.1, nonsense", "GPU-one, NVIDIA A10, 580.1, -1", "GPU-one, NVIDIA A10, 580.1, 1, extra"]) {
    expect((await Effect.runPromise(nvidiaDevices(line).pipe(Effect.either)))._tag).toBe("Left")
  }
})
