/** A chat completion request for the user's shell: POSIX quoting, or PowerShell on Windows. */
export const exampleRequest = (baseUrl: string, model: string, platform: string): string => {
  const body = JSON.stringify({ model, messages: [{ role: "user", content: "Hello" }] }).replaceAll("'", platform === "win32" ? "''" : "'\\''")
  return platform === "win32"
    ? `Invoke-RestMethod ${baseUrl}/chat/completions -Method Post -ContentType "application/json" -Body '${body}'`
    : `curl ${baseUrl}/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '${body}'`
}
