<script lang="ts">
  import type { RawInputToken } from '../types'

  interface Props {
    tokens: readonly RawInputToken[]
  }

  let { tokens }: Props = $props()

  function visibleText(text: string): string {
    if (text === '\n') return '⏎'
    if (text === '\r\n') return '⏎'
    if (text === ' ') return '·'
    if (text === '\t') return '⇥'
    if (/^\s+$/.test(text) && text.length > 1) return '␣'
    return text
  }

  function isWhitespace(text: string): boolean {
    return /^\s+$/.test(text)
  }

  let hoveredId = $state<number | null>(null)
  let tooltipX = $state(0)
  let tooltipY = $state(0)
</script>

<span class="token-strip">
  {#each tokens as token, i}
    <span
      class="token"
      class:whitespace={isWhitespace(token.text)}
      class:odd={i % 2 === 1}
      onmouseenter={(e) => {
        hoveredId = token.id
        tooltipX = e.clientX + 12
        tooltipY = e.clientY + 12
      }}
      onmouseleave={() => hoveredId = null}
      role="button"
      tabindex="0"
    >{visibleText(token.text)}</span>
  {/each}
</span>

{#if hoveredId !== null}
  <div class="tooltip" style="left: {tooltipX}px; top: {tooltipY}px;">
    <span class="id">ID: {hoveredId}</span>
  </div>
{/if}

<style>
  .token-strip {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.875rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .token {
    display: inline-block;
    cursor: pointer;
    border-radius: 2px;
    transition: filter 0.1s;
    padding: 1px 2px;
    margin: 0 1px;
    background: rgba(128, 128, 128, 0.08);
  }

  .token.odd {
    background: rgba(128, 128, 128, 0.15);
  }

  .token.whitespace {
    color: var(--text-muted);
    font-size: 0.75rem;
  }

  .token:hover {
    filter: brightness(1.25);
    background: rgba(128, 128, 128, 0.3);
  }

  .tooltip {
    position: fixed;
    z-index: 1000;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    pointer-events: none;
  }

  .id {
    font-family: monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
  }
</style>
