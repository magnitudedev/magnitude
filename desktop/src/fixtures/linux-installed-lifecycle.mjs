import assert from 'node:assert/strict';
import {join} from 'node:path';
import {mkdtemp,rm,readFile,writeFile,stat} from 'node:fs/promises';
import {spawn,execFileSync} from 'node:child_process';
import {createConnection} from 'node:net';
import {setTimeout as delay} from 'node:timers/promises';
import {_electron as electron} from 'playwright';
assert.equal(process.platform,'linux');
assert.equal(process.versions.bun,undefined,'Playwright Electron acceptance requires Node, not the Bun node shim');
const executablePath='/usr/lib/magnitude-desktop/magnitude';
const cliExecutable=process.env.MAGNITUDE_TEST_CLI_EXECUTABLE;
assert.ok(cliExecutable,'MAGNITUDE_TEST_CLI_EXECUTABLE must name the compiled CLI');
const assembled=process.env.MAGNITUDE_TEST_ASSEMBLED_DESKTOP;
if(assembled){
 for(const file of ['magnitude','resources/magnitude-service','resources/desktop-host.node','resources/magnitude-command','resources/app.asar','resources/application-icon.png','resources/trayTemplate@2x.png','resources/Magnitude-LICENSE.txt','chrome-sandbox','LICENSES.chromium.html']){
   assert.deepEqual(await readFile(join('/usr/lib/magnitude-desktop',file)),await readFile(join(assembled,file)),`Installed payload changed: ${file}`);
 }
 assert.deepEqual(await readFile('/usr/share/doc/magnitude-desktop/copyright'),await readFile(join(assembled,'LICENSE')));
 const sandbox=await stat('/usr/lib/magnitude-desktop/chrome-sandbox');
 assert.equal(sandbox.uid,0);assert.equal(sandbox.gid,0);assert.equal(sandbox.mode & 0o7777,0o4755);
 console.log('PASS installed executables, renderer, icons and licenses match the assembled app; sandbox permissions are correct');
}
const root=await mkdtemp('/tmp/mag-login-');
const inferenceInstallation=process.env.MAGNITUDE_TEST_INFERENCE_INSTALLATION;
const env={...process.env,HOME:root,XDG_CONFIG_HOME:join(root,'config'),MAGNITUDE_ICN_PATH:inferenceInstallation??join(root,'absent-engine.json'),MAGNITUDE_SHELL_ENV_INHERITED:'1'};
delete env.MAGNITUDE_DESKTOP_PATH;
delete env.MAGNITUDE_DEV_DATA_DIR;delete env.MAGNITUDE_DEV_PORT;delete env.MAGNITUDE_DESKTOP_STATE_DIR;
const endpoint=join(root,'.magnitude/desktop/application.sock');
const entry=join(env.XDG_CONFIG_HOME,'autostart/dev.magnitude.desktop');
const until=async(fn)=>{for(let i=0;i<150;i++){if(await fn())return;await delay(100)}throw Error('condition timed out')};
const request=intent=>new Promise((resolve,reject)=>{
 const socket=createConnection(endpoint);let data='';
 socket.setTimeout(5000,()=>socket.destroy(Error('Control timeout')));
 socket.on('error',reject);
 socket.on('connect',()=>socket.write(JSON.stringify({version:1,intent})+'\n'));
 socket.on('data',chunk=>{data+=chunk;if(data.includes('\n')){socket.destroy();try{resolve(JSON.parse(data.split('\n')[0]))}catch(e){reject(e)}}});
});
let app;let coldOwner;
const wm=spawn('xfwm4',['--compositor=off'],{env,stdio:'inherit'});
try {
 app=await electron.launch({chromiumSandbox:true,executablePath,env,timeout:20000});
 const page=await app.firstWindow();await page.waitForLoadState('domcontentloaded');
 await app.evaluate(({Menu})=>Menu.getApplicationMenu().items.find(i=>i.label==='View').submenu.items.find(i=>i.label==='Settings').click());
 await page.getByRole('button',{name:'Enable',exact:true}).click();
 await page.getByRole('button',{name:'Disable',exact:true}).waitFor({timeout:15000});
 const enabled=await readFile(entry,'utf8');
 assert.match(enabled,/--background/);assert.match(enabled,/Hidden=false/);
 execFileSync('desktop-file-validate',[entry]);
 await page.getByRole('button',{name:'Disable',exact:true}).click();
 await page.getByRole('button',{name:'Enable',exact:true}).waitFor({timeout:15000});
 assert.match(await readFile(entry,'utf8'),/Hidden=true/);
 await page.getByRole('button',{name:'Enable',exact:true}).click();
 await page.getByRole('button',{name:'Disable',exact:true}).waitFor({timeout:15000});
 await writeFile(entry,enabled.replace('Hidden=false','Hidden=true'));
 await page.getByRole('button',{name:'Enable',exact:true}).waitFor({timeout:15000});
 console.log('PASS packaged Settings enables/disables real XDG login entry and observes external disable');
 await page.getByRole('button',{name:'Enable',exact:true}).click();
 await page.getByRole('button',{name:'Disable',exact:true}).waitFor({timeout:15000});
 await app.evaluate(({Menu})=>setImmediate(()=>Menu.getApplicationMenu().items.find(i=>i.label==='File').submenu.items.find(i=>i.label==='Quit Magnitude').click()));
 await app.waitForEvent('close',{timeout:15000});app=undefined;
 execFileSync('gio',['launch',entry],{env,stdio:'inherit'});
 await until(async()=>{try{coldOwner=(await request('Observe')).pid;return true}catch{return false}});
 const windows=execFileSync('xprop',['-root','_NET_CLIENT_LIST'],{env,encoding:'utf8'}).match(/0x[0-9a-f]+/g)??[];
 for(const id of windows){const pid=execFileSync('xprop',['-id',id,'_NET_WM_PID'],{env,encoding:'utf8'});assert.notEqual(Number(pid.match(/= (\d+)/)?.[1]),coldOwner,'autostart must not map a window')}
 console.log('PASS real GLib desktop-entry cold launch starts owner without a visible window');
 await request('Quit');
 await until(()=>{try{process.kill(coldOwner,0);return false}catch(e){if(e.code==='ESRCH')return true;throw e}});coldOwner=undefined;
 console.log('PASS login-started owner accepts full Quit');
 const cli=args=>execFileSync(cliExecutable,args,{env,encoding:'utf8',timeout:20000});
 assert.match(cli(['service','install']),/start in the background when you log in/);
 coldOwner=(await request('Observe')).pid;
 assert.match(await readFile(entry,'utf8'),/Hidden=false/);
 if(inferenceInstallation)await until(()=>/Runtime\s+Ready/.test(cli(['service','status'])));
 const status=cli(['service','status']);
 assert.match(status,inferenceInstallation?/Runtime\s+Ready/:/Runtime\s+(Starting|Failed)/);
 if(inferenceInstallation)console.log('PASS installed desktop owns a Ready service with the supplied inference installation');
 assert.match(status,/Starts at login\s+Yes/);
 assert.match(cli(['service','stop']),/service stopped/);
 await until(()=>{try{process.kill(coldOwner,0);return false}catch(e){if(e.code==='ESRCH')return true;throw e}});coldOwner=undefined;
 assert.match(await readFile(entry,'utf8'),/Hidden=false/);
 assert.match(cli(['service','uninstall']),/removed from login startup and quit/);
 assert.match(await readFile(entry,'utf8'),/Hidden=true/);
 assert.match(cli(['service','status']),/Runtime\s+Stopped/);
 console.log('PASS compiled Linux CLI install/status/stop/uninstall uses the desktop owner and preserves stop-versus-uninstall semantics');
 app=await electron.launch({chromiumSandbox:true,executablePath,args:['--background'],env,timeout:20000});
 await (await app.firstWindow()).waitForLoadState('domcontentloaded');
 const shutdownProcess=app.process();
 const shutdownClosed=app.waitForEvent('close',{timeout:15000});
 await app.evaluate(({powerMonitor})=>powerMonitor.emit('shutdown',{preventDefault(){throw Error('System shutdown must not be vetoed')}}));
 await shutdownClosed;app=undefined;
 assert.equal(shutdownProcess.exitCode,0);
 assert.match(cli(['service','status']),/Runtime\s+Stopped/);
 console.log('PASS simulated Linux powerMonitor shutdown: no veto, exit0, owner stopped; not an OS logout test');

 {
   execFileSync('gio',['launch','/usr/share/applications/magnitude-desktop.desktop'],{env,stdio:'inherit'});
   await until(async()=>{try{coldOwner=(await request('Observe')).pid;return true}catch{return false}});
   await until(()=>{
     const ids=execFileSync('xprop',['-root','_NET_CLIENT_LIST'],{env,encoding:'utf8'}).match(/0x[0-9a-f]+/g)??[];
     return ids.some(id=>Number(execFileSync('xprop',['-id',id,'_NET_WM_PID'],{env,encoding:'utf8'}).match(/= (\d+)/)?.[1])===coldOwner);
   });
   await request('Quit');
   await until(()=>{try{process.kill(coldOwner,0);return false}catch(e){if(e.code==='ESRCH')return true;throw e}});coldOwner=undefined;
   console.log('PASS installed application-menu entry visibly opens the same owner; default CLI launch needs no path override');
 }
} finally {
 if(app)await app.close();
 if(coldOwner){await request('Quit').catch(()=>{});await until(()=>{try{process.kill(coldOwner,0);return false}catch(e){if(e.code==='ESRCH')return true;throw e}})}
 wm.kill('SIGTERM');
 await rm(root,{recursive:true,force:true});
}
