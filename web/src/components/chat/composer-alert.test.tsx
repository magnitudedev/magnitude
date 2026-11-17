import { renderToStaticMarkup } from "react-dom/server"
import { Result } from "@effect-atom/atom-react"
import { Cause } from "effect"
import { describe, expect, it } from "vitest"
import { ComposerAlert, composerError } from "./composer-alert"

const DOWN = "Magnitude service is not available"
const down = () => Result.failure(Cause.fail({ message: DOWN }))

describe("the composer's error line", () => {
  it("says once that the service is down when the chat's settings and the models both fail for it", () => {
    const error = composerError(false, [Result.initial(), Result.initial(), down(), down()])
    const html = renderToStaticMarkup(<>{error && <ComposerAlert text={error} onDismiss={() => {}} />}</>)
    expect(html.split(DOWN)).toHaveLength(2)
    expect(html.match(/role="alert"/g)).toHaveLength(1)
  })

  it("names the models' failure when the settings were read, and nothing while a request is pending", () => {
    expect(composerError(false, [Result.success({}), down()])).toBe(DOWN)
    expect(composerError(true, [down(), down()])).toBeNull()
  })

  it("says nothing more when the chat that couldn't open already says the service is down", () => {
    expect(composerError(false, [down(), down()], DOWN)).toBeNull()
    expect(composerError(false, [Result.failure(Cause.fail({ message: "Send failed" }))], DOWN)).toBe("Send failed")
  })
})
