import { useMemo, useState } from 'react';
import { TextAttributes } from '@opentui/core';
import { type ContentState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../display-types';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { StreamingMarkdownContent } from '../../markdown/markdown-content';
import { highlightFile } from '../../markdown/highlight-file';
import { isMarkdownFile, renderCodeLines } from '../../utils/file-lang';
import { BOX_CHARS } from '../../utils/ui-constants';
import { useTheme } from '../../hooks/use-theme';
import { useStreamingReveal } from '../../hooks/use-streaming-reveal';

const SHIMMER_INTERVAL_MS = 160;

export const contentDisplay = createToolDisplay<ContentState>('fileWrite', {
  render: ({ state, onFileClick }) => {
    const theme = useTheme();
    const path = state.path;
    const content = state.body ?? '';
    const isStreaming = state.phase === 'streaming' || state.phase === 'executing';
    const isDone = state.phase === 'completed';
    const isError = state.phase === 'error' || state.phase === 'interrupted' || state.phase === 'rejected';
    const [isHovered, setIsHovered] = useState(false);

    const { displayedContent, showCursor } = useStreamingReveal(content, isStreaming);
    const codeLines = useMemo(
      () => (path && !isMarkdownFile(String(path))) ? highlightFile(displayedContent, String(path), theme.syntax) : null,
      [displayedContent, path, theme.syntax],
    );

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button
          onClick={() => { if (path) onFileClick?.(path) }}
          onMouseOver={() => setIsHovered(true)}
          onMouseOut={() => setIsHovered(false)}
        >
          <box style={{ flexDirection: 'column' }}>
            <text style={{ wrapMode: 'word' }}>
              <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '✎ '}</span>
              {isDone ? (
                <>
                  <span style={{ fg: theme.foreground }}>{'Wrote '}</span>
                  <span style={{ fg: isHovered ? theme.link : theme.primary }} attributes={TextAttributes.UNDERLINE}>{String(path ?? 'file')}</span>
                  <span style={{ fg: theme.muted }}>{` (${state.lineCount} lines)`}</span>
                </>
              ) : isError ? (
                <>
                  <span style={{ fg: theme.foreground }}>{'Write '}</span>
                  <span style={{ fg: theme.muted }}>{String(path ?? '...')}</span>
                  <span style={{ fg: theme.error }}>{' · Error'}</span>
                </>
              ) : (
                <>
                  <span style={{ fg: theme.foreground }}>{'Writing '}</span>
                  <span style={{ fg: theme.muted }}>{String(path ?? '...')}</span>
                  <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
                </>
              )}
            </text>
            {isStreaming && (
              <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
                {`${state.charCount} chars · ${state.lineCount} lines`}
              </text>
            )}
            {isStreaming && content.length > 0 && (
              <box style={{
                borderStyle: 'single',
                borderColor: isHovered ? (theme.link) : (theme.border || theme.muted),
                customBorderChars: BOX_CHARS,
                height: 12,
              }}>
                <scrollbox
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
                  {codeLines === null ? (
                    <StreamingMarkdownContent content={displayedContent} showCursor={showCursor} />
                  ) : (
                    <box style={{ flexDirection: 'column' }}>
                      {codeLines.map((line, idx) => renderCodeLines(line, idx, theme.foreground))}
                    </box>
                  )}
                </scrollbox>
              </box>
            )}
          </box>
        </Button>
      </box>
    );
  },
  summary: (state) => {
    const path = state.path || 'file';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Writing ${String(path)}`;
    return `Wrote ${String(path)}`;
  },
});
