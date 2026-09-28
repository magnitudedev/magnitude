import { initializeAppearance } from "../../../web/src/stores/appearance-store"
import { registry } from './client'
const errorPhase = new URLSearchParams(location.search).get('phase')
const failRead = () => Promise.reject(new Error('PRIVATE host diagnostics'))
// Observe real renderer state without reaching into the user's filesystem or service.
window.__magnitudeDesktop = {
 getModelStorage:async()=>errorPhase === 'error' ? failRead() : ({active:'/models',path:errorPhase === 'restart-error' ? '/new-models' : '/models',source:'Default',defaultPath:'/models',pending:errorPhase === 'restart-error',warning:null}),
 getNetworkAccess:async()=>errorPhase === 'error' ? failRead() : ({enabled:false,bind:null,requireApiKey:true,apiKey:null,interfaces:[],pending:false,warning:null}),
 relaunch:async()=>failRead(),
 platform:'darwin', observe(callback:any){callback({version:1,pid:1,endpoint:'http://127.0.0.1:1',service:{_tag:'Ready',health:{service:'magnitude-acn',version:'0.0.14',revision:1,id:'fixture',pid:1,state:{_tag:'Ready'},rpcVersion:1}},owner:{_tag:'Desktop',tray:{_tag:'Registered'}}});return()=>{}},
 getAppearance:async()=>new URLSearchParams(location.search).get("theme") ?? "system",setAppearance:async()=>{},actions:()=>()=>{},presentModel:async()=>{},
}
// The acceptance entry renders App in the fixture registry, without starting the service.
window.loadingRegistry=registry
initializeAppearance(new URLSearchParams(location.search).get('theme') === 'dark' ? 'dark' : 'light')
await import('../../src/renderer')
