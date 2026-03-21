import { type BrowserActionState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../types';
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

export const browserActionDisplay = createToolDisplay<BrowserActionState>(
  ['click', 'doubleClick', 'rightClick', 'type', 'scroll', 'drag', 'navigate', 'goBack', 'switchTab', 'newTab', 'screenshot', 'evaluate'],
  {
  render: ({ state }) => {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const icon = getIcon(state.label);

    if (isRunning) {
      return (
        <box style={{ flexDirection: 'column' }}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: theme.info }}>{icon} </span>
            <span style={{ fg: theme.foreground }}>{state.label}</span>
            {state.detail ? <span style={{ fg: theme.muted }}>{state.detail}</span> : null}
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </text>
        </box>
      );
    }

    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : `${icon} `}</span>
          <span style={{ fg: theme.foreground }}>{state.label}</span>
          {state.detail ? <span style={{ fg: theme.muted }}>{state.detail}</span> : null}
          {isError && <span style={{ fg: theme.error }}>{' · Error'}</span>}
        </text>
      </box>
    );
  },
  summary: (state) => {
    const label = (state.label ?? '').trim().replace(/\s+/g, ' ');
    const detail = (state.detail ?? '').trim().replace(/\s+/g, ' ');
    if (label.length === 0) return 'Browser action';
    if (detail.length === 0) return label;
    const noSpaceBeforeDetail = /^[,.;:!?)]/.test(detail);
    const noSpaceAfterLabel = /[([]$/.test(label);
    const separator = (noSpaceBeforeDetail || noSpaceAfterLabel) ? '' : ' ';
    return `${label}${separator}${detail}`;
  },
});
