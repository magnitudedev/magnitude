<script lang="ts">
  import { logprobToBgColor } from '../logprobColors'

  export interface TokenWithLogprob {
    token: string
    logprob: number
    topLogprobs: { token: string; logprob: number }[]
  }

  interface Props {
    tokens: TokenWithLogprob[]
    onHover?: (token: TokenWithLogprob, index: number, event: MouseEvent) => void
    onLeave?: () => void
  }

  let { tokens, onHover, onLeave }: Props = $props()
</script>

<span class="token-renderer">
  {#each tokens as token, i}
    <span
      class="token"
      style="background: {logprobToBgColor(token.logprob)}; color: #000"
      onmouseenter={(e) => onHover?.(token, i, e)}
      onmouseleave={() => onLeave?.()}
      role="button"
      tabindex="0"
    >{token.token}</span>
  {/each}
</span>

<style>
  .token-renderer {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.875rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .token {
    cursor: pointer;
    border-radius: 2px;
    transition: filter 0.1s;
    padding: 1px 2px;
    margin: 0 1px;
  }

  .token:hover {
    filter: brightness(1.25);
  }
</style>