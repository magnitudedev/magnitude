import { describe, expect, it } from "vitest"
import {
  CHAT_LANGUAGES,
  CHAT_TRANSLATION_COUNTS,
  CHAT_TRANSLATION_KEYS,
  isChatLanguage,
  translateChatText,
} from "./chat-i18n"

describe("native Chat localization", () => {
  it("has a complete translation entry for every UI key in every active language", () => {
    for (const language of CHAT_LANGUAGES) {
      expect(CHAT_TRANSLATION_COUNTS[language], language).toBe(CHAT_TRANSLATION_KEYS.length)
    }
  })

  it("translates the primary Chat action", () => {
    expect(translateChatText("New chat", "es")).toBe("Nuevo chat")
    expect(translateChatText("New chat", "zh-CN")).toBe("新建聊天")
  })

  it("translates dynamic local-model errors without hiding the diagnostic", () => {
    expect(translateChatText("Could not load local models: HTTP 503", "es"))
      .toBe("No se pudieron cargar los modelos locales: HTTP 503")
  })

  it("preserves unknown diagnostics and model-specific text", () => {
    expect(translateChatText("backend-specific diagnostic", "fr")).toBe("backend-specific diagnostic")
    expect(translateChatText("qwen3.5-9b-instruct", "fr")).toBe("qwen3.5-9b-instruct")
  })

  it("accepts only the active language allowlist", () => {
    expect(isChatLanguage("es")).toBe(true)
    expect(isChatLanguage("vi")).toBe(false)
  })
})
