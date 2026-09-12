import type { LocalModel } from "@magnitudedev/sdk"
import qwen from "../../assets/brand/model-providers/qwen.svg"
import deepseek from "../../assets/brand/model-providers/deepseek.svg"
import gemma from "../../assets/brand/model-providers/gemma.png"
import liquid from "../../assets/brand/model-providers/liquid-ai.svg"
import nvidia from "../../assets/brand/model-providers/nvidia.svg"
import poolside from "../../assets/brand/model-providers/poolside.svg"
import zai from "../../assets/brand/model-providers/zai.svg"
import prism from "../../assets/brand/model-providers/prismml.svg"
import meta from "../../assets/brand/model-providers/meta.svg"

// Presentation artwork follows canonical catalog families; it does not infer model capabilities.
const families = [
  { prefix: "qwen", name: "Qwen", src: qwen, theme: "" },
  { prefix: "deepseek", name: "DeepSeek", src: deepseek, theme: "" },
  { prefix: "gemma", name: "Gemma", src: gemma, theme: "" },
  { prefix: "lfm", name: "Liquid AI", src: liquid, theme: "dark:invert" },
  { prefix: "nemotron", name: "NVIDIA", src: nvidia, theme: "" },
  { prefix: "laguna", name: "Poolside", src: poolside, theme: "" },
  { prefix: "glm", name: "Z.ai", src: zai, theme: "invert dark:invert-0" },
  { prefix: "bonsai", name: "PrismML", src: prism, theme: "invert dark:invert-0" },
  { prefix: "llama", name: "Meta", src: meta, theme: "" },
  { prefix: "muse", name: "Meta", src: meta, theme: "" },
  { prefix: "glimmer", name: "Meta", src: meta, theme: "" },
] as const

export function ModelLogo({ model, className = "size-10" }: {
  readonly model: Pick<LocalModel, "modelId" | "presentation">
  readonly className?: string
}) {
  const logo = families.find(family => model.modelId.startsWith(family.prefix))
  return logo
    ? <img src={logo.src} alt={`${logo.name} logo`} className={`shrink-0 object-contain ${className} ${logo.theme}`} />
    : <span aria-hidden="true" className={`inline-flex shrink-0 items-center justify-center font-heading text-slate-600 dark:text-slate-300 ${className}`}>{model.presentation.displayName.slice(0, 1)}</span>
}
