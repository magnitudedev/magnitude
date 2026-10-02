/**
 * A chat completion request for the user's shell: POSIX quoting, or PowerShell on Windows. From
 * another device it sends the Network access key, shown as a placeholder.
 */
export const exampleRequest = (baseUrl: string, model: string, platform: string, withKey = false): string => {
  const body = JSON.stringify({ model, messages: [{ role: "user", content: "Hello" }] }).replaceAll("'", platform === "win32" ? "''" : "'\\''")
  return platform === "win32"
    ? `Invoke-RestMethod ${baseUrl}/chat/completions -Method Post -ContentType "application/json"${withKey ? ' -Headers @{ Authorization = "Bearer YOUR_KEY" }' : ""} -Body '${body}'`
    : `curl ${baseUrl}/chat/completions \\\n${withKey ? '  -H "Authorization: Bearer YOUR_KEY" \\\n' : ""}  -H "Content-Type: application/json" \\\n  -d '${body}'`
}
