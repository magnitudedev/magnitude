import { Option } from "effect"
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
import huggingFace from "../../../assets/brand/hugging-face.svg"

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

const labForCatalogId = (modelId: string): ModelLab | undefined =>
  modelLabs.find(lab => lab.prefixes.some(prefix => modelId.startsWith(prefix)))

/** A lab named anywhere in a free-form model name, for files with no catalog lineage. */
export const labForName = (name: string): ModelLab | undefined => {
  const lower = name.toLowerCase()
  return modelLabs.find(lab => lab.prefixes.some(prefix => lower.includes(prefix)))
}

/** A catalog model's lab from its ID; a custom model's from the catalog model it varies, else its name. */
export const modelLab = (model: LocalModel): ModelLab | undefined =>
  model._tag === "Catalog"
    ? labForCatalogId(model.modelId)
    : Option.match(model.presentation.variantOf, {
      onNone: () => labForName(model.presentation.displayName),
      onSome: labForCatalogId,
    })

/** Custom artwork follows its variant family only when the displayed name still names that family. */
export const variantLab = (variantOf: Option.Option<string>, name: string): ModelLab | undefined =>
  Option.getOrUndefined(Option.flatMap(variantOf, id => Option.filter(
    Option.fromNullable(labForCatalogId(id)),
    lab => lab.prefixes.some(prefix => new RegExp(`(^|[^\\p{L}\\p{N}])${prefix}(?!\\p{L})`, "iu").test(name)),
  )))

/** Artwork is separate from the family classification used by filters and compatibility labels. */
export const modelLogoLab = (model: LocalModel): ModelLab | undefined =>
  model._tag === "Catalog" ? modelLab(model) : variantLab(model.presentation.variantOf, model.presentation.displayName)

const modelInitialName = (model: LocalModel): string =>
  model._tag === "Custom"
    ? model.origin._tag === "Added"
      ? model.origin.repository.split("/")[0]!
      : /^hf:([^/]+)\//.exec(model.modelId)?.[1] ?? model.presentation.displayName
    : model.presentation.displayName

export function LabLogo({ lab, className = "size-10" }: { readonly lab: ModelLab; readonly className?: string }) {
  return <img src={lab.src} alt={`${lab.name} logo`} className={`shrink-0 object-contain ${className} ${lab.theme}`} />
}

/** A lab's logo, or the name's initial when no lab is known. */
export function NamedLogo({ lab, name, className = "size-10" }: { readonly lab: ModelLab | undefined; readonly name: string; readonly className?: string }) {
  return lab
    ? <LabLogo lab={lab} className={className} />
    : <span aria-hidden="true" className={`inline-flex shrink-0 items-center justify-center font-heading text-slate-600 dark:text-slate-300 ${className}`}>{Array.from(name.trim())[0]?.toUpperCase()}</span>
}

export function ModelLogo({ model, className = "size-10" }: { readonly model: LocalModel; readonly className?: string }) {
  return <NamedLogo lab={modelLogoLab(model)} name={modelInitialName(model)} className={className} />
}

/** The Hugging Face mark, decorative beside text that already names Hugging Face. */
export function HuggingFaceLogo({ className = "size-4" }: { readonly className?: string }) {
  return <img src={huggingFace} alt="" aria-hidden="true" className={`shrink-0 object-contain ${className}`} />
}
