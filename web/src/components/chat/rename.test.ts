import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { SessionId } from "@magnitudedev/sdk"
import { renameOf } from "./rename"

describe("Renaming a chat", () => {
  it("renames from the title the chat had when editing began, whatever its row shows at save", () => {
    const chatId = SessionId.make("chat")
    // Editing begins at "First"; the row then refreshes to a title renamed elsewhere.
    const edit = { chatId, from: "First" }
    expect(renameOf(edit, "  Mine ")).toEqual(Option.some({ chatId, from: "First", title: "Mine" }))
    expect(renameOf(edit, "   ")).toEqual(Option.none())
  })
})
