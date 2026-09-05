import * as FileSystem from "@effect/platform/FileSystem"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import { getAgentDir, parseArgs, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent"
import { Effect, Exit, Scope } from "effect"
import { fileURLToPath } from "node:url"

export const SETUP_PROMPT = "Set up local models for me with the Magnitude CLI. Install it with `npm i -g @magnitudedev/cli` (or my package manager), then run `magnitude docs onboarding` and follow the instructions."
export const SETUP_QUESTION = "Set up local models with Magnitude now?"
export const SETUP_REMINDER = "You can set up local models anytime with `/magnitude-setup`."

/** An exclusive, profile-owned receipt: concurrent Pi launches can offer setup only once. */
export const claimSetupOffer = (agentDir: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = `${agentDir}/magnitude`
  yield* fs.makeDirectory(directory, { recursive: true })
  return yield* fs.writeFileString(`${directory}/setup-offered`, "", { flag: "wx" }).pipe(
    Effect.as(true),
    Effect.catchTag("SystemError", (error) => error.reason === "AlreadyExists" ? Effect.succeed(false) : Effect.fail(error)),
  )
})

export function canOfferSetup(ctx: ExtensionContext, argv: readonly string[]): boolean {
  if (ctx.mode !== "tui" || !ctx.isIdle() || ctx.hasPendingMessages() || ctx.ui.getEditorText().trim()) return false
  if (ctx.sessionManager.getEntries().some((entry) => entry.type === "message")) return false
  if (ctx.modelRegistry.getAll().some((model) => model.provider === "magnitude")) return false
  const args = parseArgs([...argv])
  return args.messages.length === 0 && args.fileArgs.length === 0
}

export const offerSetup = (pi: ExtensionAPI, ctx: ExtensionContext, agentDir: string, signal: AbortSignal) => Effect.gen(function* () {
  if (!(yield* claimSetupOffer(agentDir)) || signal.aborted) return
  const accepted = yield* Effect.tryPromise(() => ctx.ui.confirm(SETUP_QUESTION, "", { signal }))
  if (signal.aborted) return
  if (accepted && ctx.isIdle() && !ctx.hasPendingMessages()) pi.sendUserMessage(SETUP_PROMPT)
  else pi.sendMessage({ customType: "magnitude-setup", content: SETUP_REMINDER, display: true })
})

export function registerMagnitudeOnboarding(pi: ExtensionAPI): () => Promise<void> {
  const abort = new AbortController()
  const scope = Effect.runSync(Scope.make())
  pi.registerCommand("magnitude-setup", {
    description: "Set up local models with Magnitude",
    handler: async (_args, ctx) => {
      if (!ctx.isIdle() || ctx.hasPendingMessages()) {
        ctx.ui.notify("Wait for the current task to finish, then run /magnitude-setup.", "warning")
        return
      }
      pi.sendUserMessage(SETUP_PROMPT)
    },
  })
  pi.on("resources_discover", () => {
    // Pi loads shared/user skills first. Never introduce a duplicate named skill.
    if (parseArgs(process.argv.slice(2)).noSkills || pi.getCommands().some((command) => command.source === "skill" && command.name === "skill:magnitude")) return
    return { skillPaths: [fileURLToPath(new URL("./skills/magnitude/SKILL.md", import.meta.url))] }
  })
  pi.on("session_start", (event, ctx) => {
    if (event.reason !== "startup" || !canOfferSetup(ctx, process.argv.slice(2))) return
    // Do not await a human decision inside Pi's extension-binding lifecycle.
    // Pi must finish binding resources and its transcript while the dialog is open.
    Effect.runFork(Effect.forkIn(offerSetup(pi, ctx, getAgentDir(), abort.signal).pipe(
      Effect.provide(NodeFileSystem.layer),
      Effect.catchAll(() => Effect.sync(() => ctx.ui.notify(`Magnitude setup could not open. ${SETUP_REMINDER}`, "warning"))),
    ), scope))
  })
  return () => {
    abort.abort()
    return Effect.runPromise(Scope.close(scope, Exit.void))
  }
}
