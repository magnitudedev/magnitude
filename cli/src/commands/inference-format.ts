import { Option } from "effect"
import {
  deriveHardwareRoles,
  formatLocalInferenceBackend,
  formatMemorySize,
} from "@magnitudedev/client-common"
import type { LocalInferenceHardware } from "@magnitudedev/sdk"

export const formatContext = (tokens: number): string => tokens < 1_000
  ? String(tokens)
  : tokens % 1_024 === 0
    ? `${tokens / 1_024}K`
    : `${Math.round(tokens / 1_000)}K`

export interface LocalHardwarePresentation {
  readonly system: { readonly name: string; readonly details: readonly string[] }
  readonly accelerators: readonly { readonly name: string; readonly details: string }[]
}

export const describeLocalHardware = (
  hardware: LocalInferenceHardware,
): LocalHardwarePresentation => {
  const roles = deriveHardwareRoles(hardware)
  // With unified memory, the device in use is the system memory line's GPU.
  const [first] = roles.devices
  const unifiedGpu = roles.unifiedMemory && first?.role._tag === "InUse" ? first.role : undefined
  return {
    system: {
      name: roles.machineName,
      details: [
        `${hardware.platform === "MacOS" ? "macOS" : hardware.platform} · ${hardware.architecture === "Arm64" ? "ARM64" : "x86-64"} · ${hardware.logicalCores} logical CPU core${hardware.logicalCores === 1 ? "" : "s"}`,
        `${formatMemorySize(hardware.totalSystemMemoryBytes)} ${roles.unifiedMemory ? "unified" : "system"} memory${unifiedGpu === undefined ? "" : ` · ${formatLocalInferenceBackend(unifiedGpu.backend)} GPU acceleration · in use`}`,
      ],
    },
    accelerators: roles.devices.slice(unifiedGpu === undefined ? 0 : 1).map(({ accelerator, role, dedicatedMemoryBytes }) => ({
      name: accelerator.name.trim() || "GPU",
      details: [
        Option.match(dedicatedMemoryBytes, { onNone: () => "shares system memory", onSome: bytes => `${formatMemorySize(bytes)} VRAM` }),
        ...(role._tag === "InUse" ? [`${formatLocalInferenceBackend(role.backend)} acceleration`, "in use"] : ["available"]),
      ].join(" · "),
    })),
  }
}
