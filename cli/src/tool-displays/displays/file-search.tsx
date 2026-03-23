import { TextAttributes } from '@opentui/core';
import { createToolDisplay } from '../types';
import { type FileSearchState } from '@magnitudedev/agent/src/models';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

function summarizeInputs(state: FileSearchState): string {
  const parts = [];
  if (state.pattern) parts.push(`pattern="${state.pattern}"`);
  if (state.path) parts.push(`path="${state.path}"`);
  if (state.glob) parts.push(`glob="${state.glob}"`);
  if (state.limit !== undefined) parts.push(`limit=${state.limit}`);
  return parts.join(' ');
}

function parseMatch(match: string): { line: number; text: string } {
  const pipeIdx = match.indexOf('|');
  if (pipeIdx === -1) return { line: 0, text: match };
  const prefix = match.slice(0, pipeIdx);
  const text = match.slice(pipeIdx + 1);
  const colonIdx = prefix.indexOf(':');
  const line = colonIdx !== -1 ? parseInt(prefix.slice(0, colonIdx), 10) || 0 : 0;
  return { line, text };
}

function truncateLine(text: string, max: number): string {
  if (!text) return '';
  const firstLine = text.split('\n').find(l => l.trim() !== '') ?? '';
  if (firstLine.length > max) return firstLine.slice(0, max - 3) + '...';
  return firstLine;
}

export const fileSearchDisplay = createToolDisplay<FileSearchState>({
  render: ({ state, isExpanded, onToggle }) => {
    const theme = useTheme();
    const inputSummary = summarizeInputs(state);
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const uniqueFiles = state.fileCount;

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button onClick={onToggle}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '/ '}</span>
            {isRunning ? (
              <>
                <span style={{ fg: theme.foreground }}>{'Searching '}</span>
                <span style={{ fg: theme.muted }}>{inputSummary || '...'}</span>
                <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : isError ? (
              <>
                <span style={{ fg: theme.foreground }}>{'Searched '}</span>
                <span style={{ fg: theme.muted }}>{inputSummary}</span>
                <span style={{ fg: theme.error }}>{' · Error'}</span>
                <span style={{ fg: theme.muted }}>{` (${state.errorDetail || ''})`}</span>
              </>
            ) : (
              <>
                <span style={{ fg: theme.foreground }}>{'Searched '}</span>
                <span style={{ fg: theme.muted }}>{inputSummary}</span>
                {state.matchCount > 0 ? (
                  <>
                    <span style={{ fg: theme.info }}>{` · ${state.matchCount} ${state.matchCount === 1 ? 'match' : 'matches'} in ${uniqueFiles} ${uniqueFiles === 1 ? 'file' : 'files'}`}</span>
                    <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                      {isExpanded ? ' (collapse)' : ' (expand)'}
                    </span>
                  </>
                ) : (
                  <span style={{ fg: theme.muted }}>{' · no matches'}</span>
                )}
              </>
            )}
          </text>
        </Button>

        {isExpanded && state.matchCount > 0 && (
          <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
            {state.matches.map((match, i) => {
              const parsed = parseMatch(match.match);
              return (
                <text key={i}>
                  <span style={{ fg: theme.foreground }}>{'- '}{match.file}</span>
                  <span style={{ fg: theme.muted }}>{`:${parsed.line}`}</span>
                  <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{`  ${truncateLine(parsed.text, 60)}`}</span>
                </text>
              );
            })}
          </box>
        )}
      </box>
    );
  },
  summary: (state) => {
    const summary = summarizeInputs(state);
    const target = summary.length > 0 ? summary : 'files';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Searching ${target}`;
    return `Searched ${target}`;
  },
});
