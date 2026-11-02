import { formatStopSequences, parseStopSequences } from "./stop-sequences"
import { useState, type ReactNode } from "react"
import { Option } from "effect"
import type {
  ChatDefaultsKey,
  ChatDefaultsStatus,
  ChatRequestSettings,
} from "@magnitudedev/client-common"
import { CaretDownIcon, CaretRightIcon, CheckIcon } from "@phosphor-icons/react"
import { Input } from "../ui/input"
import { none, number, value } from "./shared"

/** Where this chat's settings stand against its model's defaults, and what can be done about it. */
export interface ChatDefaultsState {
  readonly status: ChatDefaultsStatus
  readonly changed: ReadonlySet<ChatDefaultsKey>
  readonly canSave: boolean
  readonly save: () => void
  readonly revert: () => void
  readonly reset: () => void
}

export function ChangedDot() {
  return (
    <span
      role="img"
      aria-label="Changed"
      className="inline-block size-1.5 shrink-0 rounded-full bg-blue-600 dark:bg-blue-400"
    />
  )
}

const statusText: { readonly [Status in ChatDefaultsStatus]: string } = {
  standard: "Standard settings",
  changedFromStandard: "Changed from standard",
  yours: "Your defaults",
  changedFromYours: "Changed from your defaults",
}

const link =
  "rounded text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-blue-500/30 disabled:pointer-events-none disabled:opacity-40"
const quietLink = `${link} text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100`

/** Two lines under the model name: where the settings stand, then what can be done, always in the same place. */
function DefaultsStatus({ d }: { d: ChatDefaultsState }) {
  const changed = d.status === "changedFromStandard" || d.status === "changedFromYours"
  return (
    <div className="mt-1 text-xs">
      <div className="flex h-5 items-center gap-1.5 text-slate-600 dark:text-slate-300">
        <span className="flex w-1.5 shrink-0 justify-center">
          {changed ? (
            <ChangedDot />
          ) : d.status === "yours" ? (
            <CheckIcon weight="bold" className="size-3 text-green-600 dark:text-green-400" />
          ) : (
            <span className="size-1.5 rounded-full border border-slate-400 dark:border-slate-500" />
          )}
        </span>
        <span className="truncate">{statusText[d.status]}</span>
      </div>
      <div className="flex h-5 items-center gap-3 pl-3">
        {changed ? (
          <>
            <button
              type="button"
              disabled={!d.canSave}
              onClick={d.save}
              title="Start new chats with this model from these settings"
              className={`${link} text-blue-700 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300`}
            >
              {d.status === "changedFromYours" ? "Update default" : "Save as default"}
            </button>
            <button type="button" onClick={d.revert} className={quietLink}>
              Revert
            </button>
          </>
        ) : d.status === "yours" ? (
          <button type="button" onClick={d.reset} className={quietLink}>
            Reset to standard
          </button>
        ) : (
          <span className="truncate text-slate-400 dark:text-slate-500">
            Edit anything to save it as default
          </span>
        )}
      </div>
    </div>
  )
}

function Setting({
  title,
  children,
  value: numberValue,
  changed,
}: {
  title: string
  children: ReactNode
  value?: ReactNode
  changed?: boolean
}) {
  return (
    <div className="mt-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="inline-flex items-center gap-1.5 text-[13px] font-medium">
          {title}
          {changed && <ChangedDot />}
        </span>
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

const advancedKeys: ReadonlyArray<ChatDefaultsKey> = [
  "topP",
  "topK",
  "minP",
  "repetitionPenalty",
  "presencePenalty",
  "frequencyPenalty",
  "seed",
  "stop",
]

export function ChatSettingsPanel({
  settings: s,
  patch,
  context,
  defaults: d,
}: {
  settings: ChatRequestSettings
  patch: (change: Partial<ChatRequestSettings>) => void
  context?: number
  defaults?: ChatDefaultsState
}) {
  const [advanced, setAdvanced] = useState(false)
  const [stopText, setStopText] = useState(() => formatStopSequences(s.stop))
  // Typed text stays as typed while it means these stops; a revert or another model's defaults replace it.
  const shownStopText =
    JSON.stringify(parseStopSequences(stopText)) === JSON.stringify(s.stop)
      ? stopText
      : formatStopSequences(s.stop)
  const changed = (key: ChatDefaultsKey) => d?.changed.has(key) ?? false
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
    <Setting title={label} changed={changed(key)}>
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
      {d && <DefaultsStatus d={d} />}
      <Setting title="System prompt" changed={changed("systemPrompt")}>
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
        changed={changed("temperature")}
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
          <span className="inline-flex items-center gap-1.5">
            Advanced
            {!advanced && advancedKeys.some(changed) && <ChangedDot />}
          </span>
          {advanced ? <CaretDownIcon /> : <CaretRightIcon />}
        </button>
        {advanced && (
          <>
            <Setting title="Top P" value={value(s.topP) ?? 0.95} changed={changed("topP")}>
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
            <Setting title="Stop sequences" changed={changed("stop")}>
              <textarea
                aria-label="Stop sequences"
                placeholder="One sequence per line"
                rows={3}
                value={shownStopText}
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
    </>
  )
}
