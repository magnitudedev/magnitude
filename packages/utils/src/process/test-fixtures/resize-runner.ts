#!/usr/bin/env bun

import { Effect } from "effect"
import {
  interactiveProcessExitCode,
  runInteractiveProcess,
} from "../interactive-process"

const executable = process.argv[2]
const target = process.argv[3]
if (executable === undefined || target === undefined) {
  throw new Error("resize child executable and path are required")
}

const code = interactiveProcessExitCode(await Effect.runPromise(runInteractiveProcess({
  executable,
  args: [target],
  environment: process.env,
  workingDirectory: process.cwd(),
})))
process.exit(code)
