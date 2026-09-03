/** Process-level test entrypoint; production command parsing and execution remain unchanged. */
export {}

await import("../../index")
await new Promise((resolve) => setTimeout(resolve, 0))
process.exit(process.exitCode ?? 0)
