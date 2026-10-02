// Browser acceptance fixture: real renderer and layout, controllable query/host observations.
// This module is substituted only by the test Vite server, never the desktop build.
export * from "../../../packages/client-common/src/index"
import { Atom, Registry, Result, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option } from "effect"
import { makeSetupModel } from "../../../packages/client-common/src/application/fixtures/model"
import { hardware as makeHardware } from "../hardware/fixtures"
import { useSyncExternalStore } from "react"

const params = new URLSearchParams(location.search)
let phase = params.get('phase') ?? 'loading'
let page = params.get('page') ?? 'discover'
const listeners = new Set<() => void>()
const models = ['Qwen3.6 35B-A3B','Gemma 4 26B-A4B','Nemotron 3.5 Lightning 30B-A3B','Qwen3.5 4B','Gemma 4 12B'].map((name,index) => ({...makeSetupModel(true), modelId: `${['qwen','gemma','nemotron','qwen','gemma'][index]}-${index}:gguf:q4`, presentation:{...makeSetupModel(true).presentation,displayName:name}, storageBytes:17800000000, servingState:{...makeSetupModel(false).servingState,assessment:{...makeSetupModel(false).servingState.assessment,performance:[{contextTokens:25000,estimatedTokensPerSecond:66},{contextTokens:50000,estimatedTokensPerSecond:59},{contextTokens:75000,estimatedTokensPerSecond:54},{contextTokens:100000,estimatedTokensPerSecond:49}]}}}))
const preferenceModels = models.map((model, index) => ({ ...model,
 catalogData: { ...model.catalogData, intelligence: 10 + index * 20 },
 servingState: { ...model.servingState, rankingScores: Option.some({ intelligence: 0.1 + index * 0.2, speed: 0.9 - index * 0.2, fidelity: 0.9 }) },
}))
const modelState = {models, preparation:{assessment:{complete:true,settledModels:5,totalModels:5}}}
const hardware = makeHardware('Apple M4 Max', 64, 16, [{ name: 'Apple M4 Max', memory: 64, shared: true }])
const usePhase = () => useSyncExternalStore(callback => {listeners.add(callback);return()=>listeners.delete(callback)},()=>phase)
const failureModels = () => models.map((model, index) => index !== 0 ? model : {...model, acquisitionState: phase === 'disk-error' ? {_tag:'InstallFailed',failure:{_tag:'InsufficientDiskSpace',requiredBytes:20000000000,availableBytes:5000000000,message:'private diagnostic bytes'}} : {...model.acquisitionState,residencyState:phase === 'load-pending' ? {_tag:'Requested'} : phase === 'memory-error' ? {_tag:'Failed',failure:{_tag:'LowMemory',code:'low_memory',message:'not enough memory available: model requires 32358673408 bytes plus 6871947673 bytes reserved for the system',retryable:true,requiredMemoryBytes:32358673408,systemReserveBytes:6871947673,allocationHeadroomBytes:33741848576,loadBoundaryBytes:26869900903,minimumAdditionalAvailableBytes:5488772506}} : model.acquisitionState.residencyState}})
export const useCatalogModels = () => {const value=usePhase();if (['memory-error','disk-error','load-pending'].includes(value)) return Result.success({...modelState,models:failureModels()});return value === 'loading' ? Result.initial() : value === 'error' ? Result.fail('offline') : Result.success({...modelState,preparation:{assessment:{complete:!value.startsWith("assessing"),settledModels:value==="assessing"?2:value==="assessing-more"?4:5,totalModels:5}},models:page === "models" ? models : (value === "preference" ? preferenceModels : models).map(model=>({...model,acquisitionState:value.startsWith("optimize-") ? {_tag:"Optimizing",installation:{_tag:"Resolved",installedBytes:2300000000,primaryPath:"/models/fixture.gguf",ownership:"Magnitude"},residencyState:{_tag:"Unloaded"},progress:value === "optimize-preparing" ? {stage:"preparing",completed:0,total:0,device:Option.none()} : {stage:"tuning",completed:Number(value.slice("optimize-tuning-".length)),total:400,device:Option.some({deviceId:"gpu-0",backend:"metal"})}} : value.startsWith("download-") ? {_tag:"Installing",progress:{stage:value === "download-verifying" ? "verifying" : "downloading",bytesPerSecond:value === "download-unknown" ? Option.none() : Option.some(12500000),completedBytes:value === "download-unknown" ? 0 : value === "download-complete" ? 2300000000 : 295000000,totalBytes:value === "download-unknown" ? 0 : 2300000000}} : {_tag:"NotInstalled"}}))})}
export const useLocalModels = () => {const value=usePhase();return value === "loading" ? Result.initial() : value === "error" ? Result.fail("offline") : Result.success(value === "status-ready" ? {...modelState,models:models.map((model,index)=>index === 0 ? {...model,acquisitionState:{...model.acquisitionState,residencyState:{_tag:"Ready",allocation:{contextWindowTokens:4096,memoryDomains:[{memoryDomainId:"system",modelBytes:2300000000,contextBytes:100000000,computeBytes:100000000,auxiliaryBytes:0}]}}}} : model)} : modelState,{waiting:value==="refreshing"})}
export const useLocalInferenceHardware = () => {const value=usePhase();return value === 'loading' || value === 'hardware-loading' ? Result.initial() : value === 'error' ? Result.fail('offline') : Result.success(hardware,{waiting:value==="refreshing"})}
export const useLocalModelMutations = () => ({ install(){setPhase("download-unknown")},load(){setPhase("load-pending")},stop(){},cancel(){setPhase("ready")},remove(){},dismissFailure(){setPhase("ready")} })
export const useLocalModelCommandStatus = () => ({pending:false,pendingOperations:[],failures:[]})
export const useLocalModelStopStatus = () => ({pending:false,failure:Option.none()})
// Hardware budget is intentionally independent of platform-specific fixture domain schemas.
export const targetPhysicalMemoryBytes = () => 68719476736

const registry = Registry.make()
const idle = () => Atom.keepAlive(Atom.make(Result.initial()))
const action = () => Atom.fn(() => Effect.void)
const errorPhase = params.get('phase')
const fixtureSettings = <A,>(value: A) => Atom.keepAlive(Atom.make(errorPhase === 'error' ? Result.fail('PRIVATE host diagnostics') : Result.success(value)))
export const service: any = {
 page:Atom.keepAlive(Atom.make(page)),rankingPreference:Atom.keepAlive(Atom.make(2)),
 navigate:(value:string)=>Effect.sync(()=>{page=value;registry.set(service.page,value)}),
 setRankingPreference:(value:number)=>Effect.sync(()=>registry.set(service.rankingPreference,value)),
 notices:Atom.keepAlive(Atom.make([])),dismissNotice:()=>Effect.void,quitFailed:Atom.keepAlive(Atom.make(false)),resolveQuitFailure:action(),retryService:action(),
 readAppearance:Effect.succeed('system'),saveAppearance:()=>Effect.void,clientWindow:Option.some({platform:'darwin'}),serverPlatform:Option.some('darwin'),
 application:Atom.keepAlive(Atom.make(Result.success({version:1,pid:1,endpoint:'http://127.0.0.1:1',service:{_tag:'Ready',health:{service:'magnitude-acn',version:'0.0.14',revision:1,id:'fixture',pid:1,state:{_tag:'Ready'},rpcVersion:1}},owner:{_tag:'Desktop',tray:{_tag:'Registered'}}}))),
 applicationInfo:idle(), updates:idle(), loginStartup:idle(), connections:idle(),machineIdentity:idle(),memory:idle(),
 modelStorage:fixtureSettings({active:'/models',path:errorPhase === 'restart-error' ? '/new-models' : '/models',source:'Default',defaultPath:'/models',warning:null}),
 networkAccess:fixtureSettings({enabled:false,bind:null,requireApiKey:true,apiKey:null,interfaces:[],port:10100,pending:false,warning:null}),
 relaunch:Atom.fn(()=>Effect.fail('PRIVATE host diagnostics')),chooseModelStorage:action(),resetModelStorage:action(),updateNetworkAccess:action(),regenerateNetworkApiKey:action(),refreshSettings:Effect.void,
 connect:action(),disconnect:action(),checkUpdate:action(),discardUpdate:action(),downloadUpdate:action(),restartUpdate:action(),setAutoDownload:action(),setLoginStartup:action(),
}
const session = Atom.make(Result.success(service))
const usage = Atom.keepAlive(Atom.make({result:Result.initial()}))
const client = {runtime:{atom:()=>session,fn:(f:any)=>Atom.fn((input:any)=>f(input))},Models:{GetServingUsage:()=>usage}}
export const useAgentClient = () => client
export const AgentClientProvider = ({children}:any) => children
// Renderer has its own provider; use this same registry for controlled fixture transitions.
export { registry }
const connections = ['pi','opencode','hermes','openclaw','codex','claude-code','oh-my-pi','cline'].map((id,index)=>({id,name:['Pi','OpenCode','Hermes','OpenClaw','Codex','Claude Code','Oh My Pi','Cline'][index],installed:index<4,managed:false,plugin:Option.none(),inspection:phase==='connection-error'?{_tag:'Unavailable',reason:'private diagnostic'}:{_tag:'Disconnected'},configurationFiles:[]}))
export function setPhase(value:string) {
 phase=value
 const result=(data:any)=>value==='loading'?Result.initial():value==='error'?Result.fail('offline'):Result.success(data,{waiting:value==="refreshing"})
 registry.set(service.applicationInfo,result({version:'0.0.14'}))
 registry.set(service.loginStartup,result({_tag:'Disabled'}))
 registry.set(service.machineIdentity,result({_tag:'Identified',manufacturer:'Apple Inc.',model:'Mac16,5',family:Option.none(),version:Option.none(),formFactor:'Unknown'}))
 registry.set(service.connections,result({_tag:'Ready',connections}))
 registry.set(service.updates,result({preference:{_tag:'Known',autoDownload:true},transfer:value==='update-error'?{_tag:'InstallationFailed',version:'0.1.6',message:'private installer diagnostic'}:{_tag:'Idle'},check:{_tag:'Idle'}}))
 registry.set(usage,{result:result({_tag:'Available',dailyActivity:Array.from({length:368},(_,index)=>({date:new Date(Date.UTC(2025,8,14+index)).toISOString().slice(0,10),totalTokens:index%5===0?0:Math.round((Math.sin(index*7)+1)*50000)})),requests:2,inputTokens:100,cachedInputTokens:40,outputTokens:20,totalTokens:120,cachedInputRequests:2,tokensPerSecond:80,timeToFirstTokenMs:125,incompleteRequests:0,recordingFailures:0,speedSamples:2,latencySamples:2,models:[],since:null})})
 for (const listener of listeners) listener()
}
setPhase(phase)
window.loadingFixture={setPhase,navigate:(value:string)=>{page=value;registry.set(service.page,value)}}
