import { Data, Effect, Option, Schema } from "effect"
import { InferenceObservationsSchema } from "@magnitudedev/sdk"
import { useState } from "react"
import {
  host, useValue, useQuery, useMutation, Button, Input,
  STATUSBAR_AREAS, PALETTE_AREA, type PluginContext,
} from "@hermes/plugin-sdk"

const Setup = Schema.Struct({ prompt: Schema.String })
const Result = Schema.Struct({ message: Schema.String })
const Progress = Schema.Struct({ available: Schema.Boolean, observations: InferenceObservationsSchema })
const Command = Schema.Struct({
  operation: Schema.Literal("status", "load", "stop"),
  args: Schema.String,
})

class DesktopRequestFailed extends Data.TaggedError("DesktopRequestFailed")<{
  readonly message: string
}> {}

// The host owns profile routing, authentication, and its query/mutation runtime.
// Effect validates the foreign host boundary; no renderer fetch or private bridge.
function request<A, I>(ctx: PluginContext, path: string, schema: Schema.Schema<A, I>, body?: typeof Command.Type) {
  return Effect.tryPromise({
    try: () => ctx.rest<unknown>(path, body === undefined
      ? { timeoutMs: 5000 }
      : { method: "POST", body, timeoutMs: 605000 }),
    catch: (error) => new DesktopRequestFailed({
      message: error instanceof Error ? error.message : "Magnitude backend is unavailable. Enable its agent plugin and restart the gateway.",
    }),
  }).pipe(Effect.flatMap(Schema.decodeUnknown(schema)))
}

function ModelPanel({ ctx, connectionId, profile }: {
  ctx: PluginContext; connectionId: string | null; profile: string;
}) {
  const [model, setModel] = useState("")
  const gateway = useValue(host.state.gateway)
  const setup = useQuery({
    queryKey: [ctx.source, connectionId, profile, "setup"],
    queryFn: () => Effect.runPromise(request(ctx, "/setup", Setup)),
    retry: false,
    enabled: gateway === "open",
  })
  const command = useMutation({
    mutationFn: (body: typeof Command.Type) => Effect.runPromise(Effect.gen(function* () {
      if (host.state.connectionId.get() !== connectionId || host.state.profile.get() !== profile) {
        return yield* new DesktopRequestFailed({ message: "The active backend changed. Open Magnitude again before running a model command." })
      }
      return yield* request(ctx, "/command", Result, body)
    })),
    retry: false,
  })
  return <section aria-label="Magnitude local models" style={{ padding: 16, color: "var(--ui-text-secondary)" }}>
    <h2>Magnitude local models</h2>
    <p>Backend profile: {profile}. These controls manage models on that backend's machine.</p>
    <div style={{ display: "flex", gap: 8, marginBlock: 12 }}>
      <Button disabled={command.isPending} onClick={() => command.mutate({ operation: "status", args: "" })}>Show models</Button>
      <Button disabled={command.isPending} onClick={() => command.mutate({ operation: "stop", args: "" })}>Stop model</Button>
    </div>
    <form onSubmit={(event) => { event.preventDefault(); if (model.trim() && !command.isPending) command.mutate({ operation: "load", args: model.trim() }) }}>
      <label htmlFor="magnitude-model-id">Installed model ID</label>
      <div style={{ display: "flex", gap: 8, marginBlock: 8 }}>
        <Input id="magnitude-model-id" value={model} onChange={(event) => setModel(event.target.value)} placeholder="Choose an ID from Show models" />
        <Button type="submit" disabled={!model.trim() || command.isPending}>Load model</Button>
      </div>
    </form>
    <div role="status" aria-live="polite" style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
      {command.isPending ? "Waiting for Magnitude…" : command.error ? String(command.error) : command.data?.message}
    </div>
    <h3>Set up Magnitude</h3>
    <p>Send this prompt in your Hermes conversation, or run <code>magnitude setup</code> in a terminal.</p>
    {setup.error ? <p role="alert">Enable the Magnitude agent plugin and restart the gateway to load setup instructions.</p>
      : <pre style={{ whiteSpace: "pre-wrap", userSelect: "text" }}>{setup.data?.prompt ?? "Loading setup instructions…"}</pre>}
  </section>
}

function Panel({ ctx }: { ctx: PluginContext }) {
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  return <ModelPanel key={JSON.stringify([connectionId, profile])} ctx={ctx} connectionId={connectionId} profile={profile} />
}

function Status({ ctx, open }: { ctx: PluginContext; open: () => void }) {
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const gateway = useValue(host.state.gateway)
  const sessionId = useValue(host.state.focusedStoredSessionId)
  const owner = useValue(host.state.focusedSessionOwner)
  const busy = useValue(host.state.busy)
  const owned = owner !== null && owner.connectionId === connectionId && owner.profile === profile
  const progress = useQuery<typeof Progress.Type>({
    queryKey: [ctx.source, connectionId, profile, sessionId, "progress"],
    enabled: gateway === "open" && owned && sessionId !== null,
    retry: false,
    refetchInterval: 500,
    queryFn: () => Effect.runPromise(Effect.gen(function* () {
      const currentOwner = host.state.focusedSessionOwner.get()
      if (host.state.connectionId.get() !== connectionId || host.state.profile.get() !== profile ||
          currentOwner?.connectionId !== connectionId || currentOwner.profile !== profile ||
          host.state.focusedStoredSessionId.get() !== sessionId) {
        return { available: false, observations: [] }
      }
      return yield* request(ctx, `/progress?session_id=${encodeURIComponent(sessionId!)}`, Progress)
    })),
  })
  // Query caches outlive a connection. Never present cached activity as current
  // after the backend disconnects or a refresh fails.
  const observable = owned && gateway === "open" && !progress.error && progress.data?.available === true
  const active = observable ? progress.data?.observations.filter(item => item.state === "Active") ?? [] : []
  const current = active.at(-1)
  let label = "Magnitude"
  if (current && Option.isSome(current.progress)) {
    const p = current.progress.value
    switch (p.phase) {
      case "model_loading": label += ` · Loading ${Math.round(Math.max(0, Math.min(1, p.fraction)) * 100)}%`; break
      case "prefill": label += ` · Prefill ${p.completed_tokens}/${p.total_tokens} · ${p.cached_tokens} cached`; break
      case "queued": label += " · Queued"; break
      case "preparing": label += " · Preparing"; break
      case "generating": label += " · Generating"; break
    }
  } else if (current) label += " · Requesting"
  else if (observable && !busy) {
    const last = progress.data?.observations.at(-1)
    if (last && Option.isSome(last.timings)) {
      const t = last.timings.value
      label += ` · ${t.predicted_per_second.toFixed(1)} tok/s · TTFT ${(t.time_to_first_token_ms / 1000).toFixed(2)}s`
    }
  }
  if (active.length > 1) label += ` · ${active.length} requests`
  return <button type="button" onClick={open} title="Open Magnitude local model controls"
    style={{ paddingInline: 8, color: "var(--ui-text-secondary)" }}>{label}</button>
}

export default {
  id: "magnitude",
  name: "Magnitude",
  description: "Local model controls, setup instructions, and inference status",
  register(ctx: PluginContext) {
    let closeWorkspace: (() => void) | undefined
    const open = () => {
      closeWorkspace = host.openWorkspace("magnitude:models", {
        title: "Magnitude", render: () => <Panel ctx={ctx} />,
      })
    }
    ctx.onDispose(() => closeWorkspace?.())
    ctx.registerMany([
      { id: "status", area: STATUSBAR_AREAS.right, order: 120, render: () =>
        <Status ctx={ctx} open={open} /> },
      { id: "open", area: PALETTE_AREA, data: { id: "magnitude.open", label: "Open Magnitude local models", keywords: ["local", "models", "magnitude"], run: open } },
    ])
  },
}
