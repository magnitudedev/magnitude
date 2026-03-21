import { defineDisplay } from '@magnitudedev/tools';
import { browserActionModel, type BrowserActionState } from '@magnitudedev/agent/src/models';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

function getIcon(label?: string): string {
  const key = (label ?? '').toLowerCase();
  if (key.includes('double')) return '◎◎';
  if (key.includes('right')) return '◎';
  if (key.includes('click')) return '◎';
  if (key.includes('type')) return '⌨';
  if (key.includes('scroll')) return '↕';
  if (key.includes('drag')) return '⤳';
  if (key.includes('navigate')) return '→';
  if (key.includes('back')) return '←';
  if (key.includes('switch')) return '⇥';
  if (key.includes('new tab')) return '+';
  if (key.includes('screenshot')) return '◻';
  if (key.includes('evaluate')) return '▶';
  return '◎';
}

export const browserActionDisplay = defineDisplay(browserActionModel, {
  render: ({ state }) => {
    const theme = useTheme();
    const label = state.label ?? 'Browser action';
    const detail = state.detail ? ` ${state.detail}` : '';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const icon = getIcon(label);

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : `${icon} `}</span>
        {isRunning ? (
          <>
            <span>{`${label}${detail}`}</span>
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </>
        ) : isError ? (
          <span style={{ fg: theme.error }}>{`${label}${detail} · Error`}</span>
        ) : state.phase === 'rejected' ? (
          <span>{`${label}${detail} · Rejected`}</span>
        ) : state.phase === 'interrupted' ? (
          <span>{`${label}${detail} · Interrupted`}</span>
        ) : (
          <span>{`${label}${detail}`}</span>
        )}
      </text>
    );
  },
  summary: (state) => {
    const label = state.label ?? 'Browser action';
    if (state.phase === 'streaming' || state.phase === 'executing') return label;
    if (state.phase === 'completed') return label;
    if (state.phase === 'error') return `${label} error`;
    if (state.phase === 'rejected') return `${label} rejected`;
    return `${label} interrupted`;
  },
});
