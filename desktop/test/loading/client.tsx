// Browser acceptance fixture: real renderer and layout, controllable query/host observations.
// This module is substituted only by the test Vite server, never the desktop build.
export * from "../../../packages/client-common/src/index"
import { Atom, Registry, Result, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option } from "effect"
import { DesktopSession } from "../../../packages/client-common/src/desktop/service"
import { makeSetupModel } from "../../../packages/client-common/src/desktop/fixtures/model"
import { useSyncExternalStore } from "react"

let phase = 'loading'
let page = 'discover'
const listeners = new Set<() => void>()
const models = ['Qwen3.6 35B-A3B','Gemma 4 26B-A4B','Nemotron 3.5 Lightning 30B-A3B','Qwen3.5 4B','Gemma 4 12B'].map((name,index) => ({...makeSetupModel(true), modelId: `${['qwen','gemma','nemotron','qwen','gemma'][index]}-${index}:gguf:q4`, presentation:{...makeSetupModel(true).presentation,displayName:name}, storageBytes:17800000000, servingState:{...makeSetupModel(false).servingState,assessment:{...makeSetupModel(false).servingState.assessment,performance:[{contextTokens:25000,estimatedTokensPerSecond:66},{contextTokens:50000,estimatedTokensPerSecond:59},{contextTokens:75000,estimatedTokensPerSecond:54},{contextTokens:100000,estimatedTokensPerSecond:49}]}}}))
const modelState = {models, preparation:{assessment:{complete:true,settledModels:5,totalModels:5}}}
const hardware = { platform:'darwin', processor:Option.some('Apple M4 Max'),totalSystemMemoryBytes:68719476736,accelerators:[{name:'Apple M4 Max', memoryBytes:68719476736}], memoryDomains:[] }
const usePhase = () => useSyncExternalStore(callback => {listeners.add(callback);return()=>listeners.delete(callback)},()=>phase)
export const useCatalogModels = () => {const value=usePhase();return value === 'loading' ? Result.initial() : value === 'error' ? Result.fail('offline') : Result.success({...modelState,preparation:{assessment:{complete:!value.startsWith("assessing"),settledModels:value==="assessing"?2:value==="assessing-more"?4:5,totalModels:5}},models:page === "models" ? models : models.map(model=>({...model,acquisitionState:{_tag:"NotInstalled"}}))})}
export const useLocalModels = () => {const value=usePhase();return value === "loading" ? Result.initial() : value === "error" ? Result.fail("offline") : Result.success(modelState,{waiting:value==="refreshing"})}
export const useLocalInferenceHardware = () => {const value=usePhase();return value === 'loading' || value === 'hardware-loading' ? Result.initial() : value === 'error' ? Result.fail('offline') : Result.success(hardware,{waiting:value==="refreshing"})}
export const useLocalModelMutations = () => ({ install(){},load(){},stop(){},cancel(){},remove(){},dismissFailure(){} })
export const useLocalModelCommandStatus = () => ({pending:false,failures:[]})
export const useLocalModelStopStatus = () => ({pending:false,failure:Option.none()})
// Hardware budget is intentionally independent of platform-specific fixture domain schemas.
export const targetPhysicalMemoryBytes = () => 68719476736

const registry = Registry.make()
const idle = () => Atom.keepAlive(Atom.make(Result.initial()))
const action = () => Atom.fn(() => Effect.void)
const service = {
 page:Atom.keepAlive(Atom.make(page)),rankingPreference:Atom.keepAlive(Atom.make(2)),
 navigate:(value:string)=>Effect.sync(()=>{page=value;registry.set(service.page,value)}),
 setRankingPreference:(value:number)=>Effect.sync(()=>registry.set(service.rankingPreference,value)),
 applicationInfo:idle(), updates:idle(), loginStartup:idle(), connections:idle(),machineIdentity:idle(),
 connect:action(),disconnect:action(),checkUpdate:action(),discardUpdate:action(),downloadUpdate:action(),restartUpdate:action(),setAutoDownload:action(),setLoginStartup:action(),
}
const session = Atom.make(Result.success(service))
const usage = Atom.keepAlive(Atom.make({result:Result.initial()}))
const client = {runtime:{atom:()=>session,fn:(f:any)=>Atom.fn((input:any)=>f(input).pipe(Effect.provideService(DesktopSession,service)))},Models:{GetServingUsage:()=>usage}}
export const useAgentClient = () => client
export const AgentClientProvider = ({children}:any) => children
// Renderer has its own provider; use this same registry for controlled fixture transitions.
export { registry }
const connections = ['pi','opencode','hermes','openclaw','codex','claude-code','oh-my-pi','cline'].map((id,index)=>({id,name:['Pi','OpenCode','Hermes','OpenClaw','Codex','Claude Code','Oh My Pi','Cline'][index],installed:index<4,managed:false,plugin:Option.none(),inspection:{_tag:'Disconnected'},configurationFiles:[]}))
export function setPhase(value:string) {
 phase=value
 const result=(data:any)=>value==='loading'?Result.initial():value==='error'?Result.fail('offline'):Result.success(data,{waiting:value==="refreshing"})
 registry.set(service.applicationInfo,result({version:'0.0.14'}))
 registry.set(service.loginStartup,result({_tag:'Disabled'}))
 registry.set(service.machineIdentity,result({_tag:'Identified',manufacturer:'Apple Inc.',model:'Mac16,5'}))
 registry.set(service.connections,result({_tag:'Available',connections}))
 registry.set(service.updates,result({preference:{_tag:'Known',autoDownload:true},transfer:{_tag:'Idle'},check:{_tag:'Idle'}}))
 registry.set(usage,{result:result({_tag:'Available',requests:2,inputTokens:100,cachedInputTokens:40,outputTokens:20,cachedInputRequests:2,tokensPerSecond:80,timeToFirstTokenMs:125,incompleteRequests:0,recordingFailures:0,speedSamples:2,latencySamples:2,models:[],since:null})})
 for (const listener of listeners) listener()
}
window.loadingFixture={setPhase,navigate:(value:string)=>{page=value;registry.set(service.page,value)}}
