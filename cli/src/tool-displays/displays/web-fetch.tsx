import { type WebFetchState } from '@magnitudedev/agent/src/models';
import type { CommonToolProps } from '../types';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const WEB_SEARCH_SHIMMER_MS = 450;

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + '…';
}

export const webFetchDisplay = {
  render({ state }: { state: WebFetchState } & CommonToolProps) {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';

    if (isRunning) {
      return (
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: theme.info }}>[↓] </span>
          <span style={{ fg: theme.foreground }}>{'Fetching '}</span>
          <span style={{ fg: theme.muted }}>{state.url ? truncate(state.url, 60) : '...'}</span>
          <ShimmerText text=" ..." interval={WEB_SEARCH_SHIMMER_MS} primaryColor={theme.info} />
        </text>
      );
    }

    if (isError) {
      const errorMsg = state.errorDetail ?? '';
      return (
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: theme.error }}>{'✗  '}</span>
          <span style={{ fg: theme.foreground }}>{'Fetch '}</span>
          <span style={{ fg: theme.muted }}>{truncate(state.url ?? '', 60)}</span>
          <span style={{ fg: theme.error }}>{` · Error${errorMsg ? ` (${truncate(errorMsg, 80)})` : ''}`}</span>
        </text>
      );
    }

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: theme.info }}>[↓] </span>
        <span style={{ fg: theme.foreground }}>{'Fetched '}</span>
        <span style={{ fg: theme.muted }}>{truncate(state.url ?? '', 60)}</span>
      </text>
    );
  },
  summary(state: WebFetchState): string {
    const url = state.url || 'URL';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Fetching ${url}`;
    if (state.phase === 'error') return `Fetch ${url}`;
    return `Fetched ${url}`;
  },
};
