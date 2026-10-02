import { createServer } from 'vite'
import tailwind from '@tailwindcss/vite'
import {resolve} from 'node:path'
const root=resolve(import.meta.dirname,'../../..')
const server=await createServer({configFile:false,root,plugins:[{
 name:'loading-fixture',
 configureServer(server){server.middlewares.use('/loading.html',(_req,res)=>{res.setHeader('Content-Type','text/html');res.end('<html><head><meta name="viewport" content="width=device-width, initial-scale=1"/></head><body><div id="root"></div><script type="module" src="/web/test/loading/entry.ts"></script></body></html>')})}
},tailwind()],resolve:{alias:[{find:/^@magnitudedev\/client-common$/,replacement:resolve(import.meta.dirname,'client.tsx')},{find:'@',replacement:resolve(root,'web/src')}]},server:{host:'127.0.0.1',port:6091,fs:{allow:[root]}},define:{'process.env':'{}','process.platform':'"browser"','process.arch':'"browser"','process.pid':'0','process.versions':'{}'}})
await server.listen();console.log('Loading fixture ready on 6091')
