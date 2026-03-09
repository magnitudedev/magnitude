/**
 * EditFormat — the interface each edit method must implement.
 */
export interface EditFormat {
  /** Format identifier */
  id: string

  /** Format the input file content for display to the LLM */
  formatFile(filename: string, content: string): string

  /** Edit-method-specific system instructions */
  systemInstructions(): string

  /**
   * Parse LLM response and apply edits to produce the output file content.
   * Throws on parse/apply failure.
   */
  applyResponse(response: string, originalContent: string): string | Promise<string>

}
