import {chromium} from 'playwright'
import {mkdir,writeFile} from 'node:fs/promises'
import assert from 'node:assert/strict'
const output=new URL('../../../specs/26-09-15/page-skeletons/screenshots/',import.meta.url).pathname
await mkdir(output,{recursive:true})
const browser=await chromium.launch({headless:true})
const errors=[];const results=[]
for(const theme of ['light','dark']) for(const width of [800,1120,1600]) {
 const page=await browser.newPage({viewport:{width,height:900},reducedMotion:'reduce'})
 page.on('pageerror',e=>errors.push(e.message))
 await page.goto('http://127.0.0.1:6091/loading.html')
 await page.getByRole('navigation').waitFor()
 await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme)
 for(const name of (process.argv.length>2 ? process.argv.slice(2) : ['discover','catalog','models','connections','usage','status','settings'])) {
  await page.evaluate(name=>{window.loadingFixture.setPhase('loading');window.loadingFixture.navigate(name)},name)
  await page.waitForTimeout(100)
  assert.ok(await page.locator('[data-slot="skeleton"]').count(),`${name}: skeleton missing`)
  assert.equal(await page.locator('[data-loading-region] button,[data-loading-region] input,[data-loading-region] a').count(),0,`${name}: interactive placeholder`)
  const geometry=async()=>page.locator('main').evaluate(el=>({width:el.clientWidth,scrollWidth:el.scrollWidth,sections:[...el.querySelectorAll('article,section')].map(n=>{const b=n.getBoundingClientRect();return {tag:n.tagName,label:n.getAttribute('aria-label'),x:b.x,y:b.y,width:b.width,height:b.height}})}))
  const stableFrames=async()=>page.evaluate(name=>{
   const main=document.querySelector('main'); let nodes=[];
   if(name==='usage') nodes=[...main.querySelectorAll('div.rounded-2xl')];
   if(name==='status') nodes=[...main.querySelectorAll('section')];
   if(name==='settings') nodes=[...main.querySelectorAll('h2')].map(n=>n.closest('section,[aria-busy]'));
   if(name==='catalog'||name==='models'||name==='connections') nodes=[main.querySelector('article')];
   if(name==='discover') nodes=[main.querySelector('[aria-label="Your hardware"],[aria-label="Loading your hardware"],[aria-label="Detecting your hardware"]'),main.querySelector('[aria-label="Top recommendations"] > div,[aria-label="Loading recommendations"] > div[aria-hidden] > div')];
   return nodes.filter(Boolean).map(n=>{const b=n.getBoundingClientRect();return {x:b.x,y:b.y,width:b.width,height:b.height}})
  },name)
  const before=await stableFrames()
  assert.ok(await page.locator('[data-slot="skeleton"]').evaluateAll(nodes=>nodes.every(n=>getComputedStyle(n).animationName==='none')),'Reduced motion ignored')
  const loading=await geometry()
  assert.ok(loading.scrollWidth<=loading.width,`${name} loading overflows ${width}: ${loading.scrollWidth}/${loading.width}`)
  // Capture the entire scrollable page, preserving its width and card breakpoints.
  await page.evaluate(()=>{document.querySelector('#root > div').style.height='auto';document.querySelector('#root > div').style.minHeight='100vh'})
  await page.screenshot({path:`${output}${theme}-${width}-${name}-loading.png`,fullPage:true,animations:'disabled'})
  await page.evaluate(()=>{document.querySelector('#root > div').style.height='100vh';window.loadingFixture.setPhase('loaded')})
  await page.waitForTimeout(100)
  const loaded=await geometry()
  const after=await stableFrames()
  assert.equal(before.length,after.length,`${name}: missing loaded frame`)
  for(let i=0;i<before.length;i++) for(const dimension of ['x','y','width','height']) {
   // Text wrapping is unknown before discovery; require exact stable cards elsewhere.
   const tolerance=name==='discover'?12:1
   assert.ok(Math.abs(before[i][dimension]-after[i][dimension])<=tolerance,`${name} ${width} frame ${i} ${dimension}: ${before[i][dimension]} -> ${after[i][dimension]}`)
  }
  assert.equal(await page.locator('[data-slot="skeleton"]').count(),0,`${name}: placeholders remain after loading`)
  assert.ok(loaded.scrollWidth<=loaded.width,`${name} loaded overflows ${width}: ${loaded.scrollWidth}/${loaded.width}`)
  await page.evaluate(()=>document.querySelector('#root > div').style.height='auto')
  await page.screenshot({path:`${output}${theme}-${width}-${name}-loaded.png`,fullPage:true,animations:'disabled'})
  await page.evaluate(()=>document.querySelector('#root > div').style.height='100vh')
  await page.evaluate(()=>window.loadingFixture.setPhase('refreshing'))
  await page.waitForTimeout(50)
  assert.equal(await page.locator('[data-slot="skeleton"]').count(),0,`${name}: refresh blanks content`)
  await page.evaluate(()=>window.loadingFixture.setPhase('error'))
  await page.waitForTimeout(50)
  assert.equal(await page.locator('[data-slot="skeleton"]').count(),0,`${name}: failure retains placeholders`)
  results.push({theme,width,page:name,loading,loaded})
 }
 await page.evaluate(()=>{window.loadingFixture.navigate('discover');window.loadingFixture.setPhase('hardware-loading')})
 await page.getByLabel('Loading recommendations').waitFor()
 assert.equal(await page.getByText('No fitting recommendations right now.',{exact:false}).count(),0,'Pending hardware shown as empty recommendations')
 await page.evaluate(()=>window.loadingFixture.setPhase('loaded'))
 await page.getByLabel('Top recommendations').waitFor()
 // Failure must replace placeholders, not trap the user in loading.
 await page.evaluate(()=>window.loadingFixture.setPhase('error'))
 await page.waitForTimeout(100)
 assert.equal(await page.locator('[data-slot="skeleton"]').count(),0,'Failure retains skeleton')
 await page.close()
}
await browser.close()
assert.deepEqual(errors,[])
await writeFile(`${output}geometry.json`,JSON.stringify(results,null,2))
console.log(`PASS ${results.length} page/size/theme comparisons; ${results.length*2} screenshots, no overflow or browser errors`)
