import { TextAttributes } from '@opentui/core';
import { type SpawnWorkerState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../types';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';
import { useStreamingReveal } from '../../hooks/use-streaming-reveal';
import { BOX_CHARS } from '../../utils/ui-constants';
import { violet } from '../../utils/theme';

const SHIMMER_INTERVAL_MS = 160;

export const spawnWorkerDisplay = createToolDisplay<SpawnWorkerState>({
  render: ({ state }) => {
    const theme = useTheme();
    const message = state.message ?? '';
    const workerId = state.id ?? '';
    const isStreaming = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error' || state.phase === 'rejected' || state.phase === 'interrupted';
    const isCompleted = state.phase === 'completed';

    const { displayedContent, showCursor } = useStreamingReveal(message, isStreaming);

    // Completed state: worker has started, show the same visual as WorkerStartedRow
    if (isCompleted) {
      return (
        <text>
          <span style={{ fg: violet[300] }}>{'▶ '}</span>
          <span style={{ fg: theme.muted }}>{'Worker started: '}</span>
          <span style={{ fg: theme.foreground }}>{state.id}</span>
          {state.title && <span style={{ fg: theme.muted }}>{' — '}{state.title}</span>}
        </text>
      );
    }

    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ wrapMode: 'word' }}>
          {isError ? (
            <>
              <span style={{ fg: theme.error }}>{'✗ '}</span>
              <span style={{ fg: theme.muted }}>{'Starting worker '}</span>
              {workerId && <span style={{ fg: theme.foreground }}>{workerId}</span>}
              <span style={{ fg: theme.muted }}>{' with prompt'}</span>
              <span style={{ fg: theme.error }}>{' · Error'}</span>
            </>
          ) : (
            <>
              <span style={{ fg: theme.muted }}>{'Starting worker '}</span>
              {workerId && <span style={{ fg: theme.foreground }}>{workerId}</span>}
              <span style={{ fg: theme.muted }}>{' with prompt'}</span>
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.muted} />
            </>
          )}
        </text>
        {isStreaming && displayedContent.length > 0 && (
          <box style={{
            borderStyle: 'single',
            borderColor: theme.border,
            customBorderChars: BOX_CHARS,
            height: 8,
          }}>
            <scrollbox
              onMouseScroll={(e) => e.stopPropagation()}
              stickyScroll
              stickyStart="bottom"
              scrollX={false}
              scrollbarOptions={{ visible: false }}
              verticalScrollbarOptions={{ visible: false }}
              style={{
                flexGrow: 1,
                rootOptions: { flexGrow: 1, backgroundColor: 'transparent' },
                wrapperOptions: { border: false, backgroundColor: 'transparent', paddingLeft: 1, paddingRight: 1 },
                contentOptions: { justifyContent: 'flex-start' },
              }}
            >
              <text style={{ fg: theme.muted, wrapMode: 'word' }} attributes={TextAttributes.DIM}>
                {displayedContent}
                {showCursor && <span style={{ fg: theme.info }}>{'▎'}</span>}
              </text>
            </scrollbox>
          </box>
        )}
      </box>
    );
  },
  summary: (state) => {
    const id = state.id ? ` ${state.id}` : '';
    if (state.phase === 'completed') {
      return `Worker started: ${state.id}${state.title ? ` — ${state.title}` : ''}`;
    }
    if (state.phase === 'streaming' || state.phase === 'executing') return `Starting worker${id} with prompt...`;
    if (state.phase === 'error') return `Starting worker${id} with prompt · Error`;
    if (state.phase === 'rejected') return `Starting worker${id} with prompt · Rejected`;
    if (state.phase === 'interrupted') return `Starting worker${id} with prompt · Interrupted`;
    return `Starting worker${id} with prompt`;
  },
});