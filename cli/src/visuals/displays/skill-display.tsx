import { defineDisplay } from '@magnitudedev/tools';
import { skillModel, type SkillState } from '@magnitudedev/agent/src/models';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const skillDisplay = defineDisplay(skillModel, {
  render: ({ state }) => {
    const theme = useTheme();
    const name = state.name ?? 'skill';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: theme.info }}>{'* '}</span>
        {isRunning ? (
          <>
            <span>{`Activating skill "${name}"`}</span>
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </>
        ) : state.phase === 'completed' ? (
          <span>{`Activated skill "${name}"`}</span>
        ) : isError ? (
          <span style={{ fg: theme.error }}>{`Activate skill "${name}" · Error`}</span>
        ) : state.phase === 'rejected' ? (
          <span>{`Activate skill "${name}" · Rejected`}</span>
        ) : (
          <span>{`Activate skill "${name}" · Interrupted`}</span>
        )}
      </text>
    );
  },
  summary: (state) => {
    const name = state.name ?? 'skill';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Activating skill "${name}"`;
    if (state.phase === 'completed') return `Activated skill "${name}"`;
    if (state.phase === 'error') return `Activate skill "${name}" error`;
    if (state.phase === 'rejected') return `Activate skill "${name}" rejected`;
    return `Activate skill "${name}" interrupted`;
  },
});
