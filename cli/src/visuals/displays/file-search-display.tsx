import { TextAttributes } from '@opentui/core';
import { defineDisplay } from '@magnitudedev/tools';
import { fileSearchModel, type FileSearchState } from '@magnitudedev/agent/src/models';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

function summarizeInputs(state: FileSearchState): string {
  const parts = [];
  if (state.pattern) parts.push(`pattern="${state.pattern}"`);
  if (state.path) parts.push(`path="${state.path}"`);
  if (state.glob) parts.push(`glob="${state.glob}"`);
  return parts.join(' ');
}

function parseMatch(match: string): { line: string; snippet: string } {
  const sep = match.indexOf('|');
  if (sep === -1) return { line: '?', snippet: match };
  return {
    line: match.slice(0, sep) || '?',
    snippet: match.slice(sep + 1),
  };
}

export const fileSearchDisplay = defineDisplay(fileSearchModel, {
  render: ({ state, isExpanded, onToggle }) => {
    const theme = useTheme();
    const inputSummary = summarizeInputs(state) || 'pattern="..."';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button onClick={onToggle}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '/ '}</span>
            {isRunning ? (
              <>
                <span>{`Searching ${inputSummary}`}</span>
                <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : state.phase === 'completed' ? (
              <>
                <span>{`Searched ${inputSummary} · ${state.matchCount} matches in ${state.fileCount} files`}</span>
                <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                  {isExpanded ? ' (collapse)' : ' (expand)'}
                </span>
              </>
            ) : state.phase === 'rejected' ? (
              <span>{`Search ${inputSummary} · Rejected`}</span>
            ) : state.phase === 'interrupted' ? (
              <span>{`Search ${inputSummary} · Interrupted`}</span>
            ) : (
              <span style={{ fg: theme.error }}>{`Search ${inputSummary} · Error`}</span>
            )}
          </text>
        </Button>

        {state.phase === 'completed' && isExpanded && (
          <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
            {state.matches.map((match, idx) => {
              const parsed = parseMatch(match.match);
              return (
                <text key={`${match.file}-${idx}`} style={{ wrapMode: 'word' }}>
                  <span style={{ fg: theme.foreground }}>{match.file}</span>
                  <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                    {`:${parsed.line} `}
                  </span>
                  <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{parsed.snippet}</span>
                </text>
              );
            })}
          </box>
        )}
      </box>
    );
  },
  summary: (state) => {
    const inputSummary = summarizeInputs(state) || 'pattern';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Searching ${inputSummary}`;
    if (state.phase === 'completed') return `Searched ${inputSummary}`;
    if (state.phase === 'error') return `Search ${inputSummary} error`;
    if (state.phase === 'rejected') return `Search ${inputSummary} rejected`;
    return `Search ${inputSummary} interrupted`;
  },
});
