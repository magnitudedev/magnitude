import type { LocalModel } from "@magnitudedev/sdk"
import qwen from "../../../assets/brand/model-providers/qwen.svg"
import deepseek from "../../../assets/brand/model-providers/deepseek.svg"
import gemma from "../../../assets/brand/model-providers/gemma.png"
import liquid from "../../../assets/brand/model-providers/liquid-ai.svg"
import nvidia from "../../../assets/brand/model-providers/nvidia.svg"
import poolside from "../../../assets/brand/model-providers/poolside.svg"
import zai from "../../../assets/brand/model-providers/zai.svg"
import prism from "../../../assets/brand/model-providers/prismml.svg"
import meta from "../../../assets/brand/model-providers/meta.svg"
import openbmb from "../../../assets/brand/model-providers/openbmb.svg"

export interface ModelLab {
  readonly name: string
  readonly prefixes: readonly string[]
  readonly src: string
  readonly theme: string
}

// Presentation artwork follows canonical catalog families; it does not infer model capabilities.
export const modelLabs: readonly ModelLab[] = [
  { name: "Qwen", prefixes: ["qwen"], src: qwen, theme: "" },
  { name: "DeepSeek", prefixes: ["deepseek"], src: deepseek, theme: "" },
  { name: "Gemma", prefixes: ["gemma"], src: gemma, theme: "" },
  { name: "Liquid AI", prefixes: ["lfm"], src: liquid, theme: "dark:invert" },
  { name: "NVIDIA", prefixes: ["nemotron"], src: nvidia, theme: "" },
  { name: "Poolside", prefixes: ["laguna"], src: poolside, theme: "" },
  { name: "Z.ai", prefixes: ["glm"], src: zai, theme: "invert dark:invert-0" },
  { name: "PrismML", prefixes: ["bonsai"], src: prism, theme: "invert dark:invert-0" },
  { name: "Meta", prefixes: ["llama", "muse", "glimmer"], src: meta, theme: "" },
  { name: "OpenBMB", prefixes: ["minicpm"], src: openbmb, theme: "" },
]

export const modelLab = (model: Pick<LocalModel, "modelId">): ModelLab | undefined =>
  modelLabs.find(lab => lab.prefixes.some(prefix => model.modelId.startsWith(prefix)))

export function LabLogo({ lab, className = "size-10" }: { readonly lab: ModelLab; readonly className?: string }) {
  return <img src={lab.src} alt={`${lab.name} logo`} className={`shrink-0 object-contain ${className} ${lab.theme}`} />
}

export function ModelLogo({ model, className = "size-10" }: {
  readonly model: Pick<LocalModel, "modelId" | "presentation">
  readonly className?: string
}) {
  const lab = modelLab(model)
  return lab
    ? <LabLogo lab={lab} className={className} />
    : <span aria-hidden="true" className={`inline-flex shrink-0 items-center justify-center font-heading text-slate-600 dark:text-slate-300 ${className}`}>{model.presentation.displayName.slice(0, 1)}</span>
}
