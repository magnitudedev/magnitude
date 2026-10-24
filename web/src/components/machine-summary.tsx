import { useId, useState } from "react"
import { Option } from "effect"
import type { LocalInferenceHardware } from "@magnitudedev/sdk"
import { deriveHardwareRoles } from "@magnitudedev/client-common"
import type { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import { CaretDownIcon, CaretRightIcon, LaptopIcon, DesktopTowerIcon, ComputerTowerIcon, HardDrivesIcon, MonitorIcon, CpuIcon, GraphicsCardIcon, MemoryIcon } from "@phosphor-icons/react"
import { hardwareDetails } from "./hardware-details"

const GiB = 1024 ** 3
const capacity = (bytes: number) => `${Number((bytes / GiB).toFixed(1))} GiB`
export function HardwareSummary({
  value,
  identity,
  expanded,
  onExpandedChange,
}: {
  value: LocalInferenceHardware
  identity: MachineIdentityObservation | null
  expanded?: boolean
  onExpandedChange?: (expanded: boolean) => void
}) {
  const [localOpen, setLocalOpen] = useState(false)
  const open = expanded ?? localOpen
  const setOpen = onExpandedChange ?? setLocalOpen
  const id = useId()
  const details = hardwareDetails(identity, value)
  const category = details.category.toLowerCase()
  const Icon = /laptop|tablet|mobile/.test(category)
    ? LaptopIcon
    : /server/.test(category)
    ? HardDrivesIcon
    : /mini/.test(category)
    ? ComputerTowerIcon
    : /desktop|workstation/.test(category)
    ? DesktopTowerIcon
    : MonitorIcon
  const roles = deriveHardwareRoles(value)
  const unified = roles.unifiedMemory
  const dedicated = roles.devices.flatMap(({ accelerator, dedicatedMemoryBytes }) => Option.match(dedicatedMemoryBytes, {
    onNone: () => [],
    onSome: bytes => [{ name: accelerator.name, bytes }],
  }))
  const gpuSummary = dedicated.length > 1
    ? `${dedicated.length} GPUs · ${dedicated.every(device => device.bytes === dedicated[0]!.bytes)
      ? `${capacity(dedicated[0]!.bytes)} VRAM each`
      : `${dedicated.map(device => capacity(device.bytes)).join(" / ")} VRAM`}`
    : dedicated.length === 1
    ? `${dedicated[0]!.name} · ${capacity(dedicated[0]!.bytes)} VRAM`
    : Option.getOrElse(value.processor, () => "Processor not identified")
  const summary = dedicated.length
    ? gpuSummary
    : Option.getOrElse(value.processor, () => "Processor not identified")
  const machine = Option.getOrElse(details.name, () =>
    identity?.formFactor === "Desktop"
      ? "Desktop"
      : identity?.formFactor === "Server"
      ? "Server"
      : "This computer"
  )
  const groups = details.groups.map((group) => {
    const memoryBytes = "dedicatedMemoryBytes" in group ? Option.getOrUndefined(group.dedicatedMemoryBytes) : undefined
    return {
      ...group,
      name: group.label === "Memory" ? capacity(value.totalSystemMemoryBytes) : group.name,
      Icon: group.label === "Memory" ? MemoryIcon : group.label.startsWith("GPU") ? GraphicsCardIcon : CpuIcon,
      details: group.details.map((detail, index) => index === 0 && memoryBytes !== undefined
        ? `${capacity(memoryBytes)} VRAM${detail.includes("shared system RAM") ? " + shared system RAM" : ""}`
        : detail.replace(/ \(spec\)/g, "")),
    }
  })
  return (
    <section aria-label={`Your machine: ${machine}`} className="mb-6">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-3 rounded py-1 text-left text-xs text-slate-500 outline-none hover:text-slate-700 focus-visible:ring-2 focus-visible:ring-slate-400/30 dark:text-slate-400 dark:hover:text-slate-200"
      >
        <Icon
          aria-label={`${details.category} icon`}
          className="size-4 shrink-0"
        />
        <span className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-1">
          <span>Your machine:</span>
          <span className="font-medium text-slate-700 dark:text-slate-300">
            {machine}
          </span>
          <span aria-hidden="true">·</span>
          <span>{summary}</span>
          <span aria-hidden="true">·</span>
          <span>
            {capacity(value.totalSystemMemoryBytes)}{" "}
            {unified ? "unified memory" : "RAM"}
          </span>
          {details.cpuInference && (
            <>
              <span aria-hidden="true">·</span>
              <span>CPU inference</span>
            </>
          )}
        </span>
        {open ? (
          <CaretDownIcon className="-ml-1 size-3.5 shrink-0" />
        ) : (
          <CaretRightIcon className="-ml-1 size-3.5 shrink-0" />
        )}
      </button>
      {open && (
        <div
          id={id}
          className="ml-7 mt-5 pb-1"
        >
          <div className="overflow-x-auto pb-1">
            <dl className="min-w-max space-y-3.5">
              {groups.map((group) => (
                <div key={group.label} className="flex items-center gap-4 text-xs">
                  <dt className="flex w-[76px] shrink-0 items-center gap-2 text-slate-500">
                    <group.Icon className="size-3.5 shrink-0" />
                    {group.label}
                  </dt>
                  <dd className="shrink-0 whitespace-nowrap text-[13px] font-medium text-slate-800 dark:text-slate-200">{group.name}</dd>
                  <dd className="flex items-center gap-4 whitespace-nowrap">
                    {group.details.map((detail) => (
                      <span key={detail} className="flex items-center gap-4">
                        <span aria-hidden="true" className="h-3 w-px bg-slate-300/60 dark:bg-slate-600/50" />
                        <HardwareSpec detail={detail} />
                      </span>
                    ))}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </section>
  )
}
function HardwareSpec({ detail }: { detail: string }) {
  const parts = /^(\d[\d,.]*(?:\s(?:GB\/s|GiB|GB|TB|GHz|MHz))?)\s(.+)$/.exec(detail)
  return parts ? (
    <span className="text-slate-500 dark:text-slate-400"><span className="font-medium tabular-nums text-slate-700 dark:text-slate-300">{parts[1]}</span>{" "}{parts[2]}</span>
  ) : <span className="text-slate-500 dark:text-slate-400">{detail}</span>
}
