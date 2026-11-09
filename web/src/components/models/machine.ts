import { Option } from "effect"
import type { LocalInferenceHardware } from "@magnitudedev/sdk"
import type { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import { deriveHardwareRoles } from "@magnitudedev/client-common"
import { hardwarePresentation } from "../hardware-photos"
import type { MachineView } from "./view"

/** The machine card's view of the service's hardware report and the machine's identity. */
export const machineView = (hardware: LocalInferenceHardware, identity: MachineIdentityObservation | null): MachineView => {
  const domains = new Map(hardware.memoryDomains.map(domain => [domain.memoryDomainId, domain]))
  const gpus = hardware.accelerators.flatMap(accelerator => {
    const own = accelerator.executors.flatMap(executor => executor.memoryDomainIds)
      .filter((id, index, ids) => ids.indexOf(id) === index)
      .flatMap(id => { const domain = domains.get(id); return domain?.kind === "PhysicalDevice" ? [domain] : [] })
    return own.length === 0 ? [] : [{
      name: accelerator.name,
      memoryBytes: own.reduce((sum, domain) => sum + domain.totalBytes, 0),
      serving: accelerator.executors.some(executor => executor.deviceId === hardware.activeDeviceId),
    }]
  })
  const unified = hardware.memoryDomains.find(domain => domain.kind === "UnifiedMemory")
  const presentation = hardwarePresentation(identity, gpus.map(gpu => gpu.name), hardware.processor)
  return {
    name: Option.getOrElse(presentation.name, () => deriveHardwareRoles(hardware).machineName),
    chip: Option.getOrNull(hardware.processor),
    photo: Option.getOrNull(presentation.photo),
    unifiedMemoryBytes: unified?.totalBytes ?? null,
    gpus: gpus.map(({ name, memoryBytes }) => ({ name, memoryBytes })),
    servingGpu: Math.max(0, gpus.findIndex(gpu => gpu.serving)),
    ramBytes: hardware.totalSystemMemoryBytes,
    freeDiskBytes: null,
  }
}
