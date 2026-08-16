import { Cpu, MonitorCog } from "lucide-react"
import type { ReactNode } from "react"
import {
  setAppearancePreference,
  useAppearancePreference,
  type AppearancePreference,
} from "../stores/appearance-store"

export interface SettingsPanelProps {
  readonly onOpenModels: () => void
}

const APPEARANCE_OPTIONS: readonly {
  readonly value: AppearancePreference
  readonly label: string
}[] = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
]

export function SettingsPanel({ onOpenModels }: SettingsPanelProps): ReactNode {
  const appearance = useAppearancePreference()
  return (
    <div className="settings-page">
      <section className="settings-section">
        <div className="settings-section-heading">
          <MonitorCog size={18} />
          <div>
            <h2>Appearance</h2>
            <p>
              Follow this computer’s appearance or choose a theme for this
              client.
            </p>
          </div>
        </div>
        <label className="settings-control">
          <span>Theme</span>
          <select
            value={appearance}
            onChange={(event) =>
              setAppearancePreference(
                event.target.value as AppearancePreference
              )
            }
          >
            {APPEARANCE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </section>
      <section className="settings-section">
        <div className="settings-section-heading">
          <Cpu size={18} />
          <div>
            <h2>Local inference</h2>
            <p>
              Models, downloads, runtime state, and hardware are managed in the
              Model Center.
            </p>
          </div>
        </div>
        <button type="button" className="primary-button" onClick={onOpenModels}>
          Open Model Center
        </button>
      </section>
    </div>
  )
}
