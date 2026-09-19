import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { piSession } from "../src/harnesses/pi"

const probe = (variant: string) => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-pi-protocol-" })
  const fixture = `
    const emit = value => console.log(JSON.stringify(value));
    const variant = process.argv[1];
    for await (const line of console) {
      const cmd = JSON.parse(line);
      if (cmd.type === 'get_state') {
        emit({type:'response', id:cmd.id, success:true, data:{sessionId:'fixture-session',model:{id:'fixture-model',provider:'magnitude'}}});
        continue;
      }
      emit({type:'response', id:cmd.id, success:variant !== 'rejected', error:'Rejected fixture'});
      if (variant === 'retry') emit({type:'auto_retry_start'});
      emit({type:'message_update',assistantMessageEvent:{type:'text_delta',delta:'HELLO'}});
      if (variant === 'tool-error') emit({type:'tool_execution_end',toolName:'edit',isError:true});
      emit({type:'message_end',message:{role:'assistant',provider:variant === 'provider'?'other':'magnitude',model:'fixture-model',stopReason:variant === 'length'?'length':variant === 'tool-only'?'toolUse':'stop',content:[]}});
      emit({type:'agent_end'});
      if (variant === 'premature') process.exit(0);
      emit({type:'agent_settled'});
    }
  `
  const pi = yield* piSession({ executable: process.execPath, args: ["-e", fixture, variant], cwd: root, environment: {},
    stdoutLog: join(root, "rpc.jsonl"), stderrLog: join(root, "stderr.log") }, "fixture-model")
  return yield* pi.prompt("Fixture prompt").pipe(Effect.either)
})).pipe(Effect.provide(BunContext.layer)))

test("Pi accepts a settled, streamed answer from the selected provider", async () => {
  const result = await probe("pass")
  expect(result._tag).toBe("Right")
  if (result._tag === "Right") expect(result.right).toEqual({ text: "HELLO", streamed: true, tools: [], sessionId: "fixture-session" })
})
test.each(["rejected", "retry", "tool-error", "provider", "length", "tool-only", "premature"])("Pi rejects %s instead of reporting success", async variant => {
  expect((await probe(variant))._tag).toBe("Left")
})
