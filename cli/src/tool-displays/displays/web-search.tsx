import { TextAttributes } from '@opentui/core';
import { type WebSearchState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../types';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const WEB_SEARCH_SHIMMER_MS = 450;

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + '…';
}

export const webSearchDisplay = createToolDisplay<WebSearchState>({
  render: ({ state, isExpanded, onToggle }) => {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const sources = state.sources;

    if (isRunning) {
      return (
        <Button onClick={onToggle}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: theme.info }}>[⌕] </span>
            <span style={{ fg: theme.foreground }}>{'Searching web for '}</span>
            <span style={{ fg: theme.muted }}>{`"${state.query ? truncate(state.query, 50) : '...'}"`}</span>
            <ShimmerText text=" ..." interval={WEB_SEARCH_SHIMMER_MS} primaryColor={theme.info} />
          </text>
        </Button>
      );
    }

    if (isError) {
      return (
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: theme.error }}>{'✗  '}</span>
          <span style={{ fg: theme.foreground }}>{'Searched web for '}</span>
          <span style={{ fg: theme.muted }}>{`"${state.query ? truncate(state.query, 50) : ''}"`}</span>
          <span style={{ fg: theme.error }}>{' · Error'}</span>
        </text>
      );
    }

    if (sources.length > 0) {
      return (
        <box style={{ flexDirection: 'column' }}>
          <Button onClick={onToggle}>
            <text style={{ wrapMode: 'word' }}>
              <span style={{ fg: theme.info }}>[⌕] </span>
              <span style={{ fg: theme.foreground }}>{'Searched web for '}</span>
              <span style={{ fg: theme.muted }}>{`"${truncate(state.query ?? '', 50)}"`}</span>
              <span style={{ fg: theme.info }}>{` · ${sources.length} ${sources.length === 1 ? 'source' : 'sources'}`}</span>
              <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>{isExpanded ? ' (collapse)' : ' (expand)'}</span>
            </text>
          </Button>
          {isExpanded && (
            <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
              {sources.map((src, i) => (
                <text key={i}>
                  <span style={{ fg: theme.foreground }}>{'- '}{src.title}</span>
                  <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{`: ${truncate(src.url, 60)}`}</span>
                </text>
              ))}
            </box>
          )}
        </box>
      );
    }

    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: theme.info }}>[⌕] </span>
        <span style={{ fg: theme.foreground }}>{'Searched web for '}</span>
        <span style={{ fg: theme.muted }}>{`"${truncate(state.query ?? '', 50)}"`}</span>
        <span style={{ fg: theme.muted }}>{' · No Sources Found'}</span>
      </text>
    );
  },
  summary: (state) => {
    const target = state.query ? `"${state.query}"` : 'the web';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Searching web for ${target}`;
    return `Searched web for ${target}`;
  },
});
