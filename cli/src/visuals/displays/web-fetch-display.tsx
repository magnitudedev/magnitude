import { defineDisplay } from '@magnitudedev/tools';
import { webFetchModel, type WebFetchState } from '@magnitudedev/agent/src/models';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const webFetchDisplay = defineDisplay(webFetchModel, {
  render: ({ state }) => {
    const theme = useTheme();
    const url = state.url ?? 'url';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const isRejected = state.phase === 'rejected';
    const isInterrupted = state.phase === 'interrupted';

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '↓ '}</span>
        {isRunning ? (
          <>
            <span>{`Fetching ${url}`}</span>
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </>
        ) : isError ? (
          <span style={{ fg: theme.error }}>{`Fetch ${url} · Error${state.errorDetail ? ` (${state.errorDetail})` : ''}`}</span>
        ) : isRejected ? (
          <span>{`Fetch ${url} · Rejected`}</span>
        ) : isInterrupted ? (
          <span>{`Fetch ${url} · Interrupted`}</span>
        ) : (
          <span>{`Fetched ${url}`}</span>
        )}
      </text>
    );
  },
  summary: (state) => {
    const url = state.url ?? 'url';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Fetching ${url}`;
    if (state.phase === 'completed') return `Fetched ${url}`;
    if (state.phase === 'error') return `Fetch ${url} error`;
    if (state.phase === 'rejected') return `Fetch ${url} rejected`;
    return `Fetch ${url} interrupted`;
  },
});
