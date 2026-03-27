import type { ToolState, ToolKey } from '@magnitudedev/agent';
import { getBrowserActionBaseLabel, getBrowserActionIcon } from '@magnitudedev/agent/src/tools/browser-action-visuals';
import { createToolDisplay } from '../types';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

type BrowserToolKey =
  | 'click'
  | 'doubleClick'
  | 'rightClick'
  | 'type'
  | 'scroll'
  | 'drag'
  | 'navigate'
  | 'goBack'
  | 'switchTab'
  | 'newTab'
  | 'screenshot'
  | 'evaluate'

type BrowserActionToolState = Extract<ToolState, { toolKey: BrowserToolKey }>

export const browserActionDisplay = createToolDisplay<BrowserActionToolState>({
  render: ({ state }) => {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const icon = getBrowserActionIcon(state.toolKey);
    const label = state.label ?? getBrowserActionBaseLabel(state.toolKey);

    if (isRunning) {
      return (
        <box style={{ flexDirection: 'column' }}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: theme.info }}>{icon} </span>
            <span style={{ fg: theme.foreground }}>{label}</span>
            {state.detail && (
              <span style={{ fg: theme.muted }}>
                {' '}
                {state.detail}
              </span>
            )}
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </text>
        </box>
      );
    }

    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : `${icon} `}</span>
          <span style={{ fg: theme.foreground }}>{label}</span>
          {state.detail && (
            <span style={{ fg: theme.muted }}>
              {' '}
              {state.detail}
            </span>
          )}
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
