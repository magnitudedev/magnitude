/**
 * One-shot host file reads for flows that consume a value once (pasted image
 * paths): imperative reads of the `ResolvePath` and `ReadFile` queries.
 */
import { useMemo } from "react"
import { useAtomSet } from "@effect-atom/atom-react"
import { QueryClient } from "@magnitudedev/effect-query"
import {
  Files,
  type ReadFileFormat,
  type ReadFileResult,
  type ResolvePathResult,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"

export interface FileReads {
  readonly resolvePath: (params: {
    readonly cwd: string
    readonly path: string
    readonly checkExists: boolean
  }) => Promise<ResolvePathResult>
  readonly readFile: (params: {
    readonly cwd: string
    readonly path: string
    readonly format?: ReadFileFormat
  }) => Promise<ReadFileResult>
}

export function useFileReads(): FileReads {
  const client = useAgentClient()
  const resolveAtom = useMemo(() => client.runtime.fn<Parameters<FileReads["resolvePath"]>[0]>()(
    (params) => QueryClient.fetch(Files.ResolvePath, params),
  ), [client])
  const readAtom = useMemo(() => client.runtime.fn<Parameters<FileReads["readFile"]>[0]>()(
    (params) => QueryClient.fetch(Files.ReadFile, params),
  ), [client])
  const resolvePath = useAtomSet(resolveAtom, { mode: "promise" })
  const readFile = useAtomSet(readAtom, { mode: "promise" })
  return useMemo(() => ({ resolvePath, readFile }), [resolvePath, readFile])
}
