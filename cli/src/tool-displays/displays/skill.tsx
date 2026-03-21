import { type SkillState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../types';
import { useTheme } from '../../hooks/use-theme';

export const skillDisplay = createToolDisplay<SkillState>('skill', {
  render: ({ state }) => {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isDone = !isRunning;
    const label = state.name ? `Activated skill "${state.name}"` : 'Activating skill...';

    return (
      <text>
        <span style={{ fg: isDone ? theme.primary : theme.info }}>{'* '}</span>
        <span style={{ fg: isDone ? theme.foreground : theme.muted }}>{label}</span>
      </text>
    );
  },
  summary: (state) => {
    const target = state.name ? `skill "${state.name}"` : 'skill';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Activating ${target}`;
    return `Activated ${target}`;
  },
});
