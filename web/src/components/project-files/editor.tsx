import { useMemo, useState, type ReactNode } from "react"
import Editor, { DiffEditor } from "@monaco-editor/react"
import type { ProjectFileTextSnapshot, ProjectId } from "@magnitudedev/sdk"
import { MarkdownContent } from "../markdown-content"
import { Button } from "@/components/ui/button"
import { monaco } from "@/lib/monaco"
import { useResolvedAppearance } from "@/stores/appearance-store"

const languageFor = (path: string): string => {
  const normalizedPath = path.toLowerCase()
  const fileName = normalizedPath.split("/").at(-1) ?? normalizedPath
  const languages = monaco.languages.getLanguages()
  const exact = languages.find((language) =>
    language.filenames?.some((candidate) => candidate.toLowerCase() === fileName))
  if (exact !== undefined) return exact.id
  const byExtension = languages
    .flatMap((language) => (language.extensions ?? []).map((extension) => ({
      id: language.id,
      extension: extension.toLowerCase(),
    })))
    .sort((left, right) => right.extension.length - left.extension.length)
    .find(({ extension }) => normalizedPath.endsWith(extension))
  return byExtension?.id ?? "plaintext"
}

export function ProjectTextEditor({
  snapshot,
  projectId,
  conflict,
  onSave,
  initialContent,
  onDraftChange,
}: {
  readonly snapshot: ProjectFileTextSnapshot
  readonly projectId: ProjectId
  readonly conflict: ProjectFileTextSnapshot | null
  readonly onSave: (content: string) => void
  readonly initialContent: string
  readonly onDraftChange: (content: string, dirty: boolean) => void
}): ReactNode {
  const [content, setContent] = useState(initialContent)
  const appearance = useResolvedAppearance()
  const canPreviewMarkdown = /\.(md|markdown)$/i.test(snapshot.path) && snapshot.size <= 512 * 1024
  const [mode, setMode] = useState<"preview" | "source">(
    canPreviewMarkdown ? "preview" : "source",
  )
  const uri = useMemo(() => `file:///magnitude-project/${encodeURIComponent(projectId)}/${encodeURI(snapshot.path)}`, [projectId, snapshot.path])
  const editorTheme = appearance === "dark" ? "magnitude-dark" : "magnitude-light"

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {canPreviewMarkdown ? (
        <div className="flex h-10 shrink-0 items-center border-b border-slate-200 px-2 dark:border-slate-800">
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={() => setMode("preview")} className={mode === "preview" ? "bg-slate-150 dark:bg-slate-750" : ""}>Preview</Button>
            <Button variant="ghost" size="sm" onClick={() => setMode("source")} className={mode === "source" ? "bg-slate-150 dark:bg-slate-750" : ""}>Source</Button>
          </div>
        </div>
      ) : null}
      {conflict ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="border-b border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">This file changed on disk. Review your version against the current file before saving again.</div>
          <DiffEditor
            original={conflict.content}
            modified={content}
            originalModelPath={`${uri}?conflict=original`}
            modifiedModelPath={`${uri}?conflict=modified`}
            keepCurrentOriginalModel
            keepCurrentModifiedModel
            language={languageFor(snapshot.path)}
            theme={editorTheme}
            options={{
              automaticLayout: true,
              minimap: { enabled: false },
              fontFamily: "Martian Mono",
              fontSize: 13,
              folding: false,
              glyphMargin: false,
              lineNumbersMinChars: 3,
              lineDecorationsWidth: 12,
              renderSideBySide: true,
              readOnly: false,
              wordWrap: "on",
              diffWordWrap: "on",
              wrappingIndent: "same",
              scrollbar: { horizontal: "hidden" },
            }}
            onMount={(editor) => editor.getModifiedEditor().onDidChangeModelContent(() => {
              const next = editor.getModifiedEditor().getValue()
              setContent(next)
              onDraftChange(next, next !== snapshot.content)
            })}
          />
        </div>
      ) : mode === "preview" ? (
        <div className="min-h-0 flex-1 overflow-auto px-5 py-4"><MarkdownContent content={content} className="mx-auto max-w-[760px]" skipHtml /></div>
      ) : (
        <Editor
          path={uri}
          defaultLanguage={languageFor(snapshot.path)}
          defaultValue={initialContent}
          theme={editorTheme}
          onChange={(value) => {
            const next = value ?? ""
            setContent(next)
            onDraftChange(next, next !== snapshot.content)
          }}
          onMount={(editor) => editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => onSave(editor.getValue()))}
          options={{
            automaticLayout: true,
            minimap: { enabled: false },
            fontFamily: "Martian Mono",
            fontSize: 13,
            folding: false,
            glyphMargin: false,
            lineHeight: 21,
            lineNumbersMinChars: 3,
            lineDecorationsWidth: 12,
            padding: { top: 12 },
            scrollBeyondLastLine: false,
            wordWrap: "on",
            wrappingIndent: "same",
            scrollbar: { horizontal: "hidden" },
          }}
        />
      )}
    </div>
  )
}
