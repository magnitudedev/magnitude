import { formatStopSequences, parseStopSequences } from "./stop-sequences"
import { useState, type ReactNode } from "react"
import { Option } from "effect"
import type { ChatSettings } from "@magnitudedev/sdk"
import {
  ArrowCounterClockwiseIcon,
  CaretDownIcon,
  CaretRightIcon,
} from "@phosphor-icons/react"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { none, number, value } from "./shared"
function Setting({
  title,
  children,
  value: numberValue,
}: {
  title: string
  children: ReactNode
  value?: ReactNode
}) {
  return (
    <div className="mt-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="text-[13px] font-medium">{title}</span>
        {numberValue !== undefined && (
          <span className="text-xs tabular-nums text-slate-500 dark:text-slate-400">
            {numberValue}
          </span>
        )}
      </div>
      {children}
    </div>
  )
}
export function ChatSettingsPanel({
  settings: s,
  patch,
  context,
  reset,
}: {
  settings: ChatSettings
  patch: (change: Partial<ChatSettings>) => void
  context?: number
  reset: () => void
}) {
  const [advanced, setAdvanced] = useState(false)
  const [stopText, setStopText] = useState(() => formatStopSequences(s.stop))
  // Unset temperature uses the inference service's sampling default.
  const temperature = value(s.temperature) ?? 1
  const defaults = {
    maxOutputTokens: context,
    topK: 20,
    minP: 0,
    repetitionPenalty: 1,
    presencePenalty: 0,
    frequencyPenalty: 0,
    seed: undefined,
  }
  const numeric = (
    label: string,
    key:
      | "maxOutputTokens"
      | "topK"
      | "minP"
      | "repetitionPenalty"
      | "presencePenalty"
      | "frequencyPenalty"
      | "seed",
    min: number,
    max: number,
    step = 1
  ) => (
    <Setting title={label}>
      <Input
        aria-label={label}
        type="number"
        min={min}
        max={max}
        step={step}
        placeholder={key === "seed" ? "Random" : "—"}
        disabled={key === "maxOutputTokens" && context === undefined && Option.isNone(s.maxOutputTokens)}
        value={value(s[key]) ?? defaults[key] ?? ""}
        onChange={(e) =>
          patch({
            [key]:
              e.target.value === ""
                ? none()
                : Option.some(Number(e.target.value)),
          })
        }
      />
    </Setting>
  )
  return (
    <>
      <Setting title="System prompt">
        <textarea
          aria-label="System prompt"
          rows={4}
          maxLength={65_536}
          value={s.systemPrompt}
          onChange={(e) => patch({ systemPrompt: e.target.value })}
          className="w-full resize-y rounded-md border border-slate-300 bg-white px-2.5 py-2 text-[13px] leading-5 outline-none focus:border-blue-500 dark:border-slate-700 dark:bg-slate-900"
        />
      </Setting>
      <Setting
        title="Temperature"
        value={temperature}
      >
        <input
          aria-label="Temperature"
          type="range"
          min="0"
          max="2"
          step=".1"
          value={temperature}
          onChange={(e) =>
            patch({ temperature: Option.some(Number(e.target.value)) })
          }
          className="w-full accent-blue-700"
        />
        <div className="mt-2 flex justify-between text-[11px] text-slate-500">
          <span>Precise</span>
          <span>Creative</span>
        </div>
      </Setting>
      {numeric("Max output tokens", "maxOutputTokens", 1, 2_000_000)}
      <div className="mt-6 border-t border-slate-200 pt-4 dark:border-slate-750">
        <button
          type="button"
          onClick={() => setAdvanced(!advanced)}
          aria-expanded={advanced}
          className="flex w-full items-center justify-between text-[13px] font-medium"
        >
          Advanced{advanced ? <CaretDownIcon /> : <CaretRightIcon />}
        </button>
        {advanced && (
          <>
            <Setting title="Top P" value={value(s.topP) ?? 0.95}>
              <input
                aria-label="Top P"
                type="range"
                min=".01"
                max="1"
                step=".01"
                value={value(s.topP) ?? 0.95}
                onChange={(e) =>
                  patch({ topP: Option.some(Number(e.target.value)) })
                }
                className="w-full accent-blue-700"
              />
            </Setting>
            {numeric("Top K", "topK", 0, 1 << 24)}
            {numeric("Min P", "minP", 0, 1, 0.01)}
            {numeric(
              "Repetition penalty",
              "repetitionPenalty",
              0.01,
              100,
              0.01
            )}
            {numeric("Presence penalty", "presencePenalty", -2, 2, 0.1)}
            {numeric("Frequency penalty", "frequencyPenalty", -2, 2, 0.1)}
            {numeric("Seed", "seed", 0, 0xffffffff)}
            <Setting title="Stop sequences">
              <textarea
                aria-label="Stop sequences"
                placeholder="One sequence per line"
                rows={3}
                value={stopText}
                onChange={(e) => {
                  setStopText(e.target.value)
                  patch({ stop: parseStopSequences(e.target.value) })
                }}
                className="w-full rounded-md border border-slate-300 bg-white p-2 text-[13px] dark:border-slate-700 dark:bg-slate-900"
              />
              <p className="mt-2 text-xs leading-5 text-slate-500">
                One per line. Use {"\\n"} for a newline.
              </p>
            </Setting>
          </>
        )}
      </div>
      <div className="mt-6 border-t border-slate-200 pt-5 dark:border-slate-750">
        <div className="flex items-center justify-between text-[13px]">
          <span className="font-medium">Context window</span>
          <span className="text-slate-500">
            {context === undefined ? "—" : number(context)}
          </span>
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">
          Configured when the model is loaded.
        </p>
      </div>
      <Button
        variant="outline"
        size="sm"
        className="mt-6"
        onClick={() => {
          reset()
          setStopText("")
        }}
      >
        <ArrowCounterClockwiseIcon />
        Reset to defaults
      </Button>
    </>
  )
}
