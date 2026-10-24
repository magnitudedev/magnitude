/** IME uses Enter to commit a composition; it must never submit the draft. */
export const submitsChatMessage = (key: string, shift: boolean, composing: boolean, keyCode: number) => key === "Enter" && !shift && !composing && keyCode !== 229
