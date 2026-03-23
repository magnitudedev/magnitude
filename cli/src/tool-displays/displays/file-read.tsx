import { type FileReadState } from '@magnitudedev/agent/src/models';
import type { CommonToolProps } from '../types';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const fileReadDisplay = {
  render({ state }: { state: FileReadState } & CommonToolProps) {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const lineCount = state.lineCount ?? 0;

    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '→ '}</span>
          {isRunning ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Reading '}</span>
              <span style={{ fg: theme.muted }}>{state.path || '...'}</span>
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
            </>
          ) : isError ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Read '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              <span style={{ fg: theme.error }}>{' · Error'}</span>
              <span style={{ fg: theme.muted }}>{` (${state.errorDetail || ''})`}</span>
            </>
          ) : (
            <>
              <span style={{ fg: theme.foreground }}>{'Read '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              {lineCount > 0 && (
                <span style={{ fg: theme.info }}>{` · ${lineCount} ${lineCount === 1 ? 'line' : 'lines'}`}</span>
              )}
            </>
          )}
        </text>
      </box>
    );
  },
  summary(state: FileReadState): string {
    const path = state.path || 'file';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Reading ${path}`;
    return `Read ${path}`;
  },
};
