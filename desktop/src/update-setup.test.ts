import { describe, expect, it } from "vitest"
import { UPDATE_SETUP_FAILED, UpdateSetupRefused, updateSetupReason } from "./update-setup"

describe("why updates are unavailable after setup failed", () => {
  it("keeps the reason a refusal wrote, such as a Windows build with no publisher", () => {
    expect(updateSetupReason(new UpdateSetupRefused({ reason: "The Windows update publisher is missing." }))).toBe("The Windows update publisher is missing.")
  })

  it("says only the general reason for an unexpected error, never its raw message", () => {
    const reason = updateSetupReason(new Error("ENOENT: no such file or directory, open 'C:\\Users\\me\\AppData\\update-identity.json'"))
    expect(reason).toBe(UPDATE_SETUP_FAILED)
  })
})
