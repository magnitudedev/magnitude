import { registry } from './client'
// Observe real renderer state without reaching into the user's filesystem or service.
window.__magnitudeDesktop = {
 platform:'darwin', observe(callback:any){callback({version:1,pid:1,endpoint:'http://127.0.0.1:1',service:{_tag:'Ready',health:{service:'magnitude-acn',version:'0.0.14',revision:1,id:'fixture',pid:1,state:{_tag:'Ready'},rpcVersion:1}},tray:{_tag:'Registered'}});return()=>{}},
 getAppearance:async()=>"system",setAppearance:async()=>{},actions:()=>()=>{},presentModel:async()=>{},
}
// The acceptance entry renders App in the fixture registry, without starting the service.
window.loadingRegistry=registry
await import('../../src/renderer')
