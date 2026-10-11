<script lang="ts">
  import type { DashboardExperiment, DashboardRun, DashboardRunDetail } from "@magnitudedev/inference-benchmark"
  import { onMount } from "svelte"

  let experiments: DashboardExperiment[] = []
  let runs: DashboardRun[] = []
  let selected: DashboardRunDetail | null = null
  let busy = ""
  let error = ""

  interface AnalysisView {
    trialId: string
    measuredRequests: number
    validRequests: number
    responsivenessMs?: { median?: number }
    prefillTokensPerSecond?: { median?: number }
    decodeTokensPerSecond?: { median?: number }
    memory?: { peakBytes?: number; peakDeviceBytes?: Record<string, number> }
  }
  interface EvaluationView { block: number; target: string; analyses: AnalysisView[] }

  function evaluations(result: unknown): EvaluationView[] {
    const value = result as { blocks?: Array<{ index?: number; comparison?: { results?: Array<{ target?: { id?: string }; analyses?: AnalysisView[] }> } }> } | null
    return value?.blocks?.flatMap(block => block.comparison?.results?.map(evaluation => ({
      block: block.index ?? 0,
      target: evaluation.target?.id ?? "unknown",
      analyses: evaluation.analyses ?? [],
    })) ?? []) ?? []
  }

  const metric = (value: number | undefined) => value === undefined ? "—" : value.toFixed(2)
  const memory = (analysis: AnalysisView) => {
    const bytes = analysis.memory?.peakDeviceBytes
      ? Object.values(analysis.memory.peakDeviceBytes).reduce((sum, value) => sum + value, 0)
      : analysis.memory?.peakBytes
    return bytes === undefined ? "—" : `${(bytes / 1024 ** 3).toFixed(2)} GiB`
  }

  async function json<T>(url: string, init?: RequestInit): Promise<T> {
    const response = await fetch(url, init)
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async function refresh() {
    try {
      ;[experiments, runs] = await Promise.all([
        json<DashboardExperiment[]>("/api/experiments"),
        json<DashboardRun[]>("/api/runs"),
      ])
      if (selected) selected = await json<DashboardRunDetail>(`/api/runs/${selected.run.runId}`)
      error = ""
    } catch (cause) { error = cause instanceof Error ? cause.message : String(cause) }
  }

  async function action(id: string, action: "prepare" | "runs") {
    busy = `${id}-${action}`
    try { await json(`/api/experiments/${id}/${action}`, { method: "POST" }); await refresh() }
    catch (cause) { error = cause instanceof Error ? cause.message : String(cause) }
    finally { busy = "" }
  }

  async function selectRun(run: DashboardRun) {
    selected = await json<DashboardRunDetail>(`/api/runs/${run.runId}`)
  }

  async function cancel(run: DashboardRun) {
    await json(`/api/runs/${run.runId}/cancel`, { method: "POST" })
    await refresh()
  }

  onMount(() => {
    refresh()
    const interval = setInterval(refresh, 1500)
    return () => clearInterval(interval)
  })
</script>

<svelte:head><title>Inference Benchmarks</title></svelte:head>

<main class="min-h-screen p-8">
  <header class="mb-8 flex items-end justify-between border-b border-zinc-800 pb-5">
    <div><div class="text-xs uppercase tracking-[.28em] text-emerald-400">Magnitude</div><h1 class="mt-1 text-3xl font-semibold">Inference Benchmarks</h1></div>
    <div class="text-sm text-zinc-500">Filesystem-backed · local machine</div>
  </header>
  {#if error}<div class="mb-5 rounded border border-red-800 bg-red-950/40 p-3 text-sm text-red-300">{error}</div>{/if}

  <section class="mb-10">
    <h2 class="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-400">Experiments</h2>
    <div class="grid grid-cols-2 gap-4">
      {#each experiments as experiment}
        <article class="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5">
          <div class="flex justify-between gap-4"><div><h3 class="font-medium">{experiment.title}</h3><div class="mt-1 font-mono text-xs text-zinc-500">{experiment.id}</div></div><span class:!text-emerald-400={experiment.prepared} class="text-xs text-amber-400">{experiment.prepared ? "prepared" : "not prepared"}</span></div>
          <div class="mt-4 grid grid-cols-4 gap-2 text-xs text-zinc-400">
            <div><span class="block text-zinc-600">Profile</span>{experiment.profile}</div>
            <div><span class="block text-zinc-600">Context</span>{experiment.requestPolicy.contextTokensPerSequence.toLocaleString()}</div>
            <div><span class="block text-zinc-600">Sequences</span>{experiment.requestPolicy.parallelSequences}</div>
            <div><span class="block text-zinc-600">Blocks</span>{experiment.execution.blocks} · {experiment.execution.variantOrder}</div>
          </div>
          <div class="my-4 space-y-2">{#each experiment.variants as variant}<div class="rounded bg-zinc-800 px-3 py-2 text-xs"><div>{variant.id} · {variant.engine} · {variant.artifact.quantization}</div><div class="mt-1 truncate font-mono text-zinc-500" title={`${variant.artifact.repository}@${variant.artifact.revision}`}>{variant.artifact.repository}@{variant.artifact.revision.slice(0, 12)}</div></div>{/each}</div>
          <div class="flex gap-2"><button class="rounded border border-zinc-700 px-3 py-2 text-sm hover:bg-zinc-800" on:click={() => action(experiment.id, "prepare")} disabled={!!busy}>Prepare</button><button class="rounded bg-emerald-500 px-3 py-2 text-sm font-medium text-black disabled:opacity-40" on:click={() => action(experiment.id, "runs")} disabled={!experiment.prepared || !!busy}>Start run</button></div>
        </article>
      {/each}
    </div>
  </section>

  <section class="grid grid-cols-[420px_1fr] gap-5">
    <div><h2 class="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-400">Runs</h2><div class="overflow-hidden rounded-lg border border-zinc-800">{#each runs as run}<button class="block w-full border-b border-zinc-800 p-4 text-left hover:bg-zinc-900" on:click={() => selectRun(run)}><div class="flex justify-between"><span class="font-mono text-xs">{run.runId}</span><span class:text-emerald-400={run.state === "completed"} class:text-amber-400={run.state === "running"} class="text-xs">{run.state}</span></div><div class="mt-2 text-sm text-zinc-400">{run.experimentId}</div></button>{/each}</div></div>
    <div><h2 class="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-400">Run detail</h2>{#if selected}<article class="rounded-lg border border-zinc-800 bg-zinc-900/40 p-5"><div class="mb-4 flex items-center justify-between"><div><div class="font-mono text-sm">{selected.run.runId}</div><div class="text-xs text-zinc-500">{selected.run.startedAt}</div></div>{#if selected.run.state === "running"}<button class="rounded border border-red-800 px-3 py-2 text-sm text-red-300" on:click={() => cancel(selected!.run)}>Cancel</button>{/if}</div>
      {#if selected.result}
        <h3 class="mb-2 text-xs uppercase tracking-wider text-zinc-500">Measurements</h3>
        {#each evaluations(selected.result) as evaluation}
          <div class="mb-4 overflow-hidden rounded border border-zinc-800"><div class="bg-zinc-800/60 px-3 py-2 text-xs font-medium">Block {evaluation.block} · {evaluation.target}</div><table class="w-full text-xs"><thead class="text-zinc-500"><tr><th class="p-2 text-left">Trial</th><th>Measured</th><th>Valid</th><th>TTFT ms</th><th>Prefill tok/s</th><th>Decode tok/s</th><th>Peak</th></tr></thead><tbody>{#each evaluation.analyses as analysis}<tr class="border-t border-zinc-800"><td class="p-2 font-mono">{analysis.trialId}</td><td class="text-center">{analysis.measuredRequests}</td><td class="text-center">{analysis.validRequests}</td><td class="text-center">{metric(analysis.responsivenessMs?.median)}</td><td class="text-center">{metric(analysis.prefillTokensPerSecond?.median)}</td><td class="text-center">{metric(analysis.decodeTokensPerSecond?.median)}</td><td class="text-center">{memory(analysis)}</td></tr>{/each}</tbody></table></div>
        {/each}
      {/if}
      <h3 class="mb-2 mt-5 text-xs uppercase tracking-wider text-zinc-500">Live events</h3><pre class="max-h-72 overflow-auto rounded bg-black/40 p-3 text-xs text-zinc-300">{selected.events.map(event => JSON.stringify(event)).join("\n")}</pre>
      <details class="mt-4"><summary class="text-xs uppercase tracking-wider text-zinc-500">Resolved manifest</summary><pre class="mt-2 max-h-96 overflow-auto rounded bg-black/40 p-3 text-xs text-zinc-300">{JSON.stringify(selected.manifest, null, 2)}</pre></details>
      {#if selected.result}<details class="mt-4"><summary class="text-xs uppercase tracking-wider text-zinc-500">Raw result</summary><pre class="mt-2 max-h-96 overflow-auto rounded bg-black/40 p-3 text-xs text-zinc-300">{JSON.stringify(selected.result, null, 2)}</pre></details>{/if}
    </article>{:else}<div class="rounded-lg border border-dashed border-zinc-800 p-12 text-center text-zinc-600">Select a run</div>{/if}</div>
  </section>
</main>
