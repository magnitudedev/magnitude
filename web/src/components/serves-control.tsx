import type { ModelServes } from "@magnitudedev/sdk"
import { SegmentedControl } from "./ui/segmented-control"

const options = [{ value: "chat", label: "Chat" }, { value: "decisions", label: "Decisions" }] as const

/** Which kind of model a model list shows: chat models or decision models. */
export function ServesControl({ value, onChange }: { value: ModelServes; onChange: (value: ModelServes) => void }) {
  return <SegmentedControl label="Model kind" options={options} value={value} onChange={onChange} />
}
