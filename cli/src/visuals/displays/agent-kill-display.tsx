import { type AgentKillState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../display-types';
import { useTheme } from '../../hooks/use-theme';

export const agentKillDisplay = createToolDisplay<AgentKillState>('agentKill', {
  render: ({ state }) => {
    const theme = useTheme();
    const label = state.agentId ? `Dismissed agent "${state.agentId}"` : 'Dismissing agent...';

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: theme.error }}>{'x '}</span>
        <span style={{ fg: theme.foreground }}>{label}</span>
      </text>
    );
  },
  summary: (state) => {
    const target = state.agentId ? `agent "${state.agentId}"` : 'agent';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Dismissing ${target}`;
    return `Dismissed ${target}`;
  },
});
