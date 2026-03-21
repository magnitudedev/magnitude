import { type AgentCreateState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../display-types';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

export const agentCreateDisplay = createToolDisplay<AgentCreateState>('agentCreate', {
  render: ({ state }) => {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const label = state.agentId ? `Started agent "${state.agentId}"` : 'Starting agent...';

    if (isRunning) {
      return (
        <text>
          <span style={{ fg: theme.info }}>{'> '}</span>
          <ShimmerText text={label} primaryColor={theme.info} />
        </text>
      );
    }

    return (
      <text>
        <span style={{ fg: isError ? theme.error : theme.success }}>{'> '}</span>
        <span style={{ fg: theme.foreground }}>{label}</span>
      </text>
    );
  },
  summary: (state) => {
    const target = state.agentId ? `agent "${state.agentId}"` : 'agent';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Starting ${target}`;
    if (state.phase === 'error') return `Start ${target}`;
    return `Started ${target}`;
  },
});
