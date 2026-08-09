import { Cpu, Info } from "lucide-react"
import type { ReactNode } from "react"

export interface SettingsPanelProps {
  readonly onOpenModels: () => void
}

export function SettingsPanel({ onOpenModels }: SettingsPanelProps): ReactNode {
  return (
    <div className="settings-page">
      <section className="settings-section">
        <div className="settings-section-heading">
          <Cpu size={18} />
          <div>
            <h2>Local inference</h2>
            <p>Models, downloads, runtime state, and hardware are managed in the Model Center.</p>
          </div>
        </div>
        <button type="button" className="primary-button" onClick={onOpenModels}>
          Open Model Center
        </button>
      </section>
      <section className="settings-section">
        <div className="settings-section-heading">
          <Info size={18} />
          <div>
            <h2>About this client</h2>
            <p>
              The browser and desktop applications render authoritative state from the local
              Magnitude daemon. Model operations continue when this window reconnects.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
