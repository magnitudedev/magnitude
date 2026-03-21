import { defineDisplay } from '@magnitudedev/tools';
import { agentCreateModel, type AgentCreateState } from '@magnitudedev/agent/src/models';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const agentCreateDisplay = defineDisplay(agentCreateModel, {
  render: ({ state }) => {
    const theme = useTheme();
    const id = state.agentId ?? 'agent';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isErrorLike = state.phase === 'error' || state.phase === 'rejected' || state.phase === 'interrupted';

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: isErrorLike ? theme.error : theme.info }}>{isErrorLike ? '✗ ' : '> '}</span>
        {isRunning ? (
          <>
            <span>{`Starting agent "${id}"`}</span>
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </>
        ) : state.phase === 'completed' ? (
          <span>{`Started agent "${id}"`}</span>
        ) : state.phase === 'error' ? (
          <span style={{ fg: theme.error }}>{`Start agent "${id}" · Error`}</span>
        ) : state.phase === 'rejected' ? (
          <span>{`Start agent "${id}" · Rejected`}</span>
        ) : (
          <span>{`Start agent "${id}" · Interrupted`}</span>
        )}
      </text>
    );
  },
  summary: (state) => {
    const id = state.agentId ?? 'agent';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Starting agent "${id}"`;
    if (state.phase === 'completed') return `Started agent "${id}"`;
    if (state.phase === 'error') return `Start agent "${id}" error`;
    if (state.phase === 'rejected') return `Start agent "${id}" rejected`;
    return `Start agent "${id}" interrupted`;
  },
});
