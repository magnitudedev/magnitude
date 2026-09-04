import { Effect } from "effect"

// Native host packages remain external; the private FSM helper is bundled, never published as a dependency.
await Effect.runPromise(Effect.tryPromise(() => Bun.build({
  entrypoints: [new URL("../extensions/magnitude.ts", import.meta.url).pathname],
  outdir: new URL("../dist", import.meta.url).pathname,
  target: "node",
  format: "esm",
  external: ["effect", "@magnitudedev/integration-protocol", "@earendil-works/pi-ai", "@earendil-works/pi-ai/*", "@earendil-works/pi-coding-agent", "@earendil-works/pi-tui"],
})).pipe(Effect.flatMap((result) => result.success
  ? Effect.void
  : Effect.fail(new Error(result.logs.map(String).join("\n"))))))
