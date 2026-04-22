<script lang="ts">
  import { logprobToColor, logprobToPercent } from '../logprobColors'

  export interface TokenWithLogprob {
    token: string
    logprob: number
    topLogprobs: { token: string; logprob: number }[]
  }

  interface Props {
    token: TokenWithLogprob
    x: number
    y: number
  }

  let { token, x, y }: Props = $props()

  function barWidth(lp: number) {
    return Math.max(2, Math.exp(lp) * 100)
  }

  let sorted = $derived(
    [...(token.topLogprobs || [])].sort((a, b) => b.logprob - a.logprob).slice(0, 8)
  )
</script>

<div class="tooltip" style="left: {x}px; top: {y}px;">
  <div class="header">
    <span class="token-text">"{token.token}"</span>
    <span class="prob">{logprobToPercent(token.logprob)}</span>
  </div>
  <div class="alternatives">
    {#each sorted as alt}
      <div class="alt-row" class:current={alt.token === token.token}>
        <span class="alt-token">{JSON.stringify(alt.token)}</span>
        <div class="bar-wrap">
          <div class="bar" style="width: {barWidth(alt.logprob)}%; background: {logprobToColor(alt.logprob)}"></div>
        </div>
        <span class="alt-prob">{logprobToPercent(alt.logprob)}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .tooltip {
    position: fixed;
    z-index: 1000;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 0.75rem;
    min-width: 220px;
    max-width: 320px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    pointer-events: none;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  .token-text {
    font-family: monospace;
    font-size: 0.875rem;
    color: var(--text-primary);
  }

  .prob {
    font-size: 0.75rem;
    color: var(--accent-green);
    font-weight: 600;
  }

  .alternatives {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .alt-row {
    display: grid;
    grid-template-columns: 80px 1fr 45px;
    gap: 0.5rem;
    align-items: center;
    font-size: 0.75rem;
  }

  .alt-row.current {
    background: rgba(59, 130, 246, 0.1);
    border-radius: 3px;
    padding: 1px 2px;
  }

  .alt-token {
    font-family: monospace;
    color: var(--text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .bar-wrap {
    background: var(--bg-tertiary);
    border-radius: 2px;
    height: 6px;
    overflow: hidden;
  }

  .bar {
    height: 100%;
    border-radius: 2px;
    transition: width 0.2s;
  }

  .alt-prob {
    color: var(--text-muted);
    text-align: right;
  }
</style>