import { defineDisplay } from '@magnitudedev/tools';
import { agentKillModel, type AgentKillState } from '@magnitudedev/agent/src/models';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const agentKillDisplay = defineDisplay(agentKillModel, {
  render: ({ state }) => {
    const theme = useTheme();
    const id = state.agentId ?? 'agent';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: theme.error }}>{'x '}</span>
        {isRunning ? (
          <>
            <span>{`Killing agent "${id}"`}</span>
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </>
        ) : state.phase === 'completed' ? (
          <span>{`Killed agent "${id}"`}</span>
        ) : state.phase === 'error' ? (
          <span style={{ fg: theme.error }}>{`Kill agent "${id}" · Error`}</span>
        ) : state.phase === 'rejected' ? (
          <span>{`Kill agent "${id}" · Rejected`}</span>
        ) : (
          <span>{`Kill agent "${id}" · Interrupted`}</span>
        )}
      </text>
    );
  },
  summary: (state) => {
    const id = state.agentId ?? 'agent';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Killing agent "${id}"`;
    if (state.phase === 'completed') return `Killed agent "${id}"`;
    if (state.phase === 'error') return `Kill agent "${id}" error`;
    if (state.phase === 'rejected') return `Kill agent "${id}" rejected`;
    return `Kill agent "${id}" interrupted`;
  },
});
