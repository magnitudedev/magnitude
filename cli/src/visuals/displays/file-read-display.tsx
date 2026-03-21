import { defineDisplay } from '@magnitudedev/tools';
import { fileReadModel } from '@magnitudedev/agent/src/models';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const fileReadDisplay = defineDisplay(fileReadModel, {
  render: ({ state }) => {
    const theme = useTheme();
    const path = state.path ?? 'file';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const isRejected = state.phase === 'rejected';
    const isInterrupted = state.phase === 'interrupted';

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '→ '}</span>
        {isRunning ? (
          <>
            <span>{`Reading ${path}`}</span>
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </>
        ) : isError ? (
          <span style={{ fg: theme.error }}>{`Read ${path} · Error${state.errorDetail ? ` (${state.errorDetail})` : ''}`}</span>
        ) : isRejected ? (
          <span>{`Read ${path} · Rejected`}</span>
        ) : isInterrupted ? (
          <span>{`Read ${path} · Interrupted`}</span>
        ) : (
          <span>{`Read ${path} · ${state.lineCount ?? 0} lines`}</span>
        )}
      </text>
    );
  },
  summary: (state) => {
    const path = state.path ?? 'file';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Reading ${path}`;
    if (state.phase === 'completed') return `Read ${path}`;
    if (state.phase === 'error') return `Read ${path} error`;
    if (state.phase === 'rejected') return `Read ${path} rejected`;
    return `Read ${path} interrupted`;
  },
});
