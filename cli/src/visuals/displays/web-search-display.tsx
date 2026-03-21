import { TextAttributes } from '@opentui/core';
import { defineDisplay } from '@magnitudedev/tools';
import { webSearchModel, type WebSearchState } from '@magnitudedev/agent/src/models';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const webSearchDisplay = defineDisplay(webSearchModel, {
  render: ({ state, isExpanded, onToggle }) => {
    const theme = useTheme();
    const query = state.query ?? '';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button onClick={onToggle}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '⌕ '}</span>
            {isRunning ? (
              <>
                <span>{`Searching web for "${query}"`}</span>
                <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : state.phase === 'completed' ? (
              <>
                <span>{`Searched web for "${query}" · ${state.sources.length} source(s)`}</span>
                <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                  {isExpanded ? ' (collapse)' : ' (expand)'}
                </span>
              </>
            ) : state.phase === 'rejected' ? (
              <span>{`Search web for "${query}" · Rejected`}</span>
            ) : state.phase === 'interrupted' ? (
              <span>{`Search web for "${query}" · Interrupted`}</span>
            ) : (
              <span style={{ fg: theme.error }}>{`Search web for "${query}" · Error`}</span>
            )}
          </text>
        </Button>

        {state.phase === 'completed' && isExpanded && (
          <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
            {state.sources.map((source, idx) => (
              <text key={`${source.url}-${idx}`} style={{ wrapMode: 'word' }}>
                <span>{'• '}</span>
                <span style={{ fg: theme.foreground }}>{source.title}</span>
                <span>{' - '}</span>
                <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{source.url}</span>
              </text>
            ))}
          </box>
        )}
      </box>
    );
  },
  summary: (state) => {
    const query = state.query ?? '';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Searching web for "${query}"`;
    if (state.phase === 'completed') return `Searched web for "${query}"`;
    if (state.phase === 'error') return `Search web for "${query}" error`;
    if (state.phase === 'rejected') return `Search web for "${query}" rejected`;
    return `Search web for "${query}" interrupted`;
  },
});
