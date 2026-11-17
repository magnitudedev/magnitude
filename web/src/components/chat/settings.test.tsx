import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import { expect, it } from "vitest"
import { ChatSettingsFields } from "./settings"

it("shows a model's own output limit as the default for max output tokens, not its context", () => {
  const settings = {
    systemPrompt: "", temperature: Option.none(), maxOutputTokens: Option.none(), topP: Option.none(), topK: Option.none(), minP: Option.none(),
    repetitionPenalty: Option.none(), presencePenalty: Option.none(), frequencyPenalty: Option.none(), seed: Option.none(), stop: [],
    model: Option.none(), reasoningEffort: Option.none(),
  }
  const html = renderToStaticMarkup(<ChatSettingsFields settings={settings} patch={() => {}} context={262_144} maxOutput={32_768} changed={new Set()} />)
  expect(html).toMatch(/aria-label="Max output tokens"[^>]*value="32768"/)
})
