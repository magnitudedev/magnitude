import { Effect, Schema } from "effect"
import type { Page } from "playwright"
import { desktopAutomation as automation } from "../../../desktop/src/automation"
import { AssertionFailure } from "./domain"

export const DownloadProgress = Schema.Struct({ completedBytes: Schema.Int.pipe(Schema.positive()), totalBytes: Schema.Int.pipe(Schema.positive()) }).pipe(
  Schema.filter(progress => progress.completedBytes < progress.totalBytes))
export interface DesktopDownloads {
  readonly begin: (model: string) => Effect.Effect<void, AssertionFailure>
  readonly transferring: (model: string) => Effect.Effect<typeof DownloadProgress.Type, AssertionFailure>
  readonly failed: (model: string) => Effect.Effect<void, AssertionFailure>
  readonly complete: (model: string) => Effect.Effect<void, AssertionFailure>
  readonly absent: (model: string) => Effect.Effect<void, AssertionFailure>
}
const fail = (message: string) => new AssertionFailure({ message })
const action = <A>(label: string, run: () => Promise<A>) => Effect.tryPromise({ try: run,
  catch: error => fail(`${label}: ${error instanceof Error ? error.message.slice(0, 1800) : "Playwright failed"}`) })

/** Observe rendered acquisition facts, never button text, colors or screen position. */
export const playwrightDownloads = (page: Page): DesktopDownloads => {
  const card = (model: string) => page.getByTestId(automation.model(model))
  const wait = (model: string, states: readonly string[], timeout: number) => page.waitForFunction(({ id, states }) => {
    const state = document.querySelector(`[data-testid="${CSS.escape(id)}"]`)?.getAttribute("data-acquisition-state")
    return state !== null && state !== undefined && states.includes(state) ? state : false
  }, { id: automation.model(model), states }, { timeout })
  return {
    begin: model => action("Start model download", async () => {
      const previous = await card(model).getAttribute("data-acquisition-state")
      await card(model).getByTestId(automation.modelDownload).click({ timeout: 120000 })
      // A retry must leave the previous failed occurrence; a fresh immediate failure is terminal.
      const states = ["Installing", "Installed", "UpdateAvailable", ...(previous === "InstallFailed" ? [] : ["InstallFailed"])]
      const state = await wait(model, states, 60000).catch(async error => {
        const alerts = (await card(model).getByRole("alert").allTextContents()).join("; ")
        throw new Error(alerts || (error instanceof Error ? error.message : "Download did not start"))
      })
      try { if (await state.jsonValue() === "InstallFailed") throw new Error((await card(model).getByRole("alert").allTextContents()).join("; ") || "Model download failed immediately") }
      finally { await state.dispose() }
    }),
    transferring: model => action("Observe an incomplete model transfer", async () => {
      const handle = await page.waitForFunction(({ id, progressId }) => {
        const card = document.querySelector(`[data-testid="${CSS.escape(id)}"]`)
        const state = card?.getAttribute("data-acquisition-state")
        if (["Installed", "UpdateAvailable", "InstallFailed"].includes(state ?? "")) return { terminal: state }
        const progress = card?.querySelector(`[data-testid="${progressId}"]`)
        const completedBytes = Number(progress?.getAttribute("data-download-completed-bytes"))
        const totalBytes = Number(progress?.getAttribute("data-download-total-bytes"))
        return progress?.getAttribute("data-download-stage") === "downloading" && completedBytes > 0 && completedBytes < totalBytes
          ? { completedBytes, totalBytes } : false
      }, { id: automation.model(model), progressId: automation.modelDownloadProgress }, { timeout: 120000 })
      try { return await handle.jsonValue() } finally { await handle.dispose() }
    }).pipe(Effect.flatMap(value => Schema.decodeUnknown(DownloadProgress)(value).pipe(Effect.mapError(() => fail("Download did not expose positive incomplete byte progress before interruption"))))),
    failed: model => action("Observe interrupted download failure", async () => {
      const state = await wait(model, ["InstallFailed", "Installed", "UpdateAvailable"], 300000)
      try { if (await state.jsonValue() !== "InstallFailed") throw new Error("Download completed before an interruption was observed") }
      finally { await state.dispose() }
      await card(model).getByRole("alert").first().waitFor()
    }),
    complete: model => action("Observe complete model acquisition", async () => {
      const state = await wait(model, ["InstallFailed", "Installed", "UpdateAvailable"], 30 * 60000)
      try { if (await state.jsonValue() === "InstallFailed") throw new Error((await card(model).getByRole("alert").allTextContents()).join("; ")) }
      finally { await state.dispose() }
      await card(model).and(page.locator('[data-model-installed="true"]')).waitFor()
    }),
    absent: model => action("Observe removed model", async () => {
      const state = await wait(model, ["NotInstalled", "RemoveFailed"], 60000)
      try { if (await state.jsonValue() !== "NotInstalled") throw new Error("Model removal failed") }
      finally { await state.dispose() }
    }),
  }
}
