import type { FitPlacementPhrase } from "@magnitudedev/client-common"
import type { HardwarePhoto } from "../hardware-photos"

/** Where part of a model sits while it runs: fast memory, host RAM beside a discrete GPU, or the disk it streams from. */
export type PlacementTierKind = "Device" | "Unified" | "Host" | "Disk" | "Lookup"

export interface PlacementTier {
  readonly kind: PlacementTierKind
  readonly bytes: number
}

export type PlacementPhrase = FitPlacementPhrase

/** One quant of a model as this machine would run it. */
export interface QuantView {
  readonly key: string
  readonly bits: number
  readonly publisher: string | null
  readonly name: string
  readonly repository: string | null
  readonly sizeBytes: number
  readonly downloadBytes: number
  readonly drafterBytes: number
  readonly fits: boolean
  readonly measuring: boolean
  readonly speed: number | null
  readonly phrase: PlacementPhrase | null
  readonly placement: readonly PlacementTier[]
  readonly servedContext: number | null
  readonly quality: string | null
}

export interface MachineGpu {
  readonly name: string
  readonly memoryBytes: number
}

export interface MachineView {
  readonly name: string
  readonly chip: string | null
  readonly photo: HardwarePhoto | null
  readonly unifiedMemoryBytes: number | null
  readonly gpus: readonly MachineGpu[]
  readonly servingGpu: number
  readonly ramBytes: number
  readonly freeDiskBytes: number | null
}
