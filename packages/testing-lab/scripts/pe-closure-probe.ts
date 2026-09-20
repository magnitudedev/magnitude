/** Native qualification: inspect real OS files and controlled corrupt copies; never execute them. */
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { FileSystem } from "@effect/platform"
import { Config, Console, Effect, Schema } from "effect"
import { PeDependencyClosure, verifyPeDependencyClosure } from "../src/pe-dependency-closure"
import { ProcessExecutorLive } from "../src/process"
import { win32 } from 'node:path'
const Observation = Schema.Union(
  Schema.Struct({ label: Schema.Literal("x64", "x86", "owned-library", "runtime-search-path"), closure: PeDependencyClosure }),
  Schema.Struct({ label: Schema.Literal("missing-import", "wrong-architecture"), message: Schema.String }),
)
const Report = Schema.Struct({ ok: Schema.Literal(true), reports: Schema.Array(Observation) })
const main=Effect.scoped(Effect.gen(function*(){
 const fs=yield* FileSystem.FileSystem
 if (process.platform !== 'win32') throw new Error('This probe requires native Windows')
 const root=yield* fs.makeTempDirectoryScoped({prefix:'lab-pe-closure-'})
 const systemRoot=yield* Config.string('SystemRoot')
 const inspector=yield* Config.string('LAB_DEPENDENCIES_EXECUTABLE')
 const reports: (typeof Observation.Type)[]=[]
 for(const [label,system] of [['x64','System32'],['x86','SysWOW64']] as const){
  const owned=win32.join(root,label)
  yield* fs.makeDirectory(owned)
  const file=win32.join(owned,'candidate.exe')
  yield* fs.copyFile(win32.join(systemRoot,system,'cmd.exe'),file)
  const report=yield* verifyPeDependencyClosure({root:owned,roots:[file],inspector,systemRoot,cuda:false,ownedSearchPaths:[]})
  if(!report.edges.length || report.edges.some(e=>e.resolution.kind!=='system'))throw new Error('Incorrect native OS boundary')
  reports.push({label,closure:report})
  if(label==='x64'){
   const bytes=yield* fs.readFile(file)
   const original=new TextEncoder().encode('api-ms-win-crt-string-l1-1-0.dll')
   const modified=new TextEncoder().encode('lab-ms-win-crt-string-l1-1-0.dll')
   const content=Buffer.from(bytes),at=content.indexOf(original)
   if(at<0||content.indexOf(original,at+1)>=0)throw new Error('Expected unique real import name')
   content.set(modified,at)
   yield* fs.writeFile(file,content)
   const rejected=yield* verifyPeDependencyClosure({root:owned,roots:[file],inspector,systemRoot,cuda:false,ownedSearchPaths:[]}).pipe(Effect.either)
   if(rejected._tag!=='Left'||!('message'in rejected.left)||!rejected.left.message.includes('unresolved'))throw new Error('Missing import was not rejected as unresolved')
   reports.push({label:'missing-import',message:rejected.left.message})
   const bundled=win32.join(owned,'lab-ms-win-crt-string-l1-1-0.dll')
   yield* fs.copyFile(win32.join(systemRoot,'System32','ucrtbase.dll'),bundled)
   const closure=yield* verifyPeDependencyClosure({root:owned,roots:[file],inspector,systemRoot,cuda:false,ownedSearchPaths:[]})
   if(closure.files.length!==2||!closure.edges.some(e=>e.resolution.kind==='owned'))throw new Error('Owned DLL was not traversed')
   reports.push({label:'owned-library',closure})
   const runtime=win32.join(owned,'runtime')
   yield* fs.makeDirectory(runtime)
   yield* fs.rename(bundled,win32.join(runtime,win32.basename(bundled)))
   const runtimeClosure=yield* verifyPeDependencyClosure({root:owned,roots:[file],inspector,systemRoot,cuda:false,ownedSearchPaths:[runtime]})
   if(runtimeClosure.files.length!==2)throw new Error('Product runtime search path was not resolved')
   reports.push({label:'runtime-search-path',closure:runtimeClosure})
   yield* fs.remove(runtime,{recursive:true})

   yield* fs.copyFile(win32.join(systemRoot,'SysWOW64','ucrtbase.dll'),bundled)
   const wrong=yield* verifyPeDependencyClosure({root:owned,roots:[file],inspector,systemRoot,cuda:false,ownedSearchPaths:[]}).pipe(Effect.either)
   if(wrong._tag!=='Left'||!('message'in wrong.left)||!/(unresolved|architecture)/.test(wrong.left.message))throw new Error('Wrong architecture DLL was not rejected')
   reports.push({label:'wrong-architecture',message:wrong.left.message})

  }
 }
 yield* Schema.encode(Schema.parseJson(Report))({ok:true,reports}).pipe(Effect.flatMap(Console.log))
}))
BunRuntime.runMain(main.pipe(Effect.provide([BunContext.layer,ProcessExecutorLive])))
