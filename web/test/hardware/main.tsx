import { createRoot } from "react-dom/client"
import { Option } from "effect"
import { HardwareSummary, HardwarePhotograph } from "../../src/discovery-visuals"
import { hardwarePhotoInventory } from "../../src/hardware-photos"
import { hardwareScenarios, hardware, identity } from "./fixtures"
import coverage from "../../../assets/hardware/coverage.json"
import facts from "../../../assets/hardware/facts.json"
import "../../../web/src/styles/tailwind.css"
import { injectPaletteCssVars } from "../../../web/src/styles/palette-css-vars"
injectPaletteCssVars()
const dark = new URLSearchParams(location.search).has("dark")
document.documentElement.dataset.theme = dark ? "dark" : "light"
const unknowns = [
  { label: "Long unknown names — ellipsis and full name on hover", identity: identity("Example vendor", "Unrecognized custom workstation with an unusually long manufacturer model identifier 1234567890", "Desktop"), value: hardware("Unrecognized processor with an unusually long model name and additional firmware identification text", 64, 16, [{name: "Unrecognized accelerator with an unusually long vendor and product model identifier", memory: 16}]) },
  { label: "Unknown hardware — no enclosure identity", identity: null, value: hardware("Unrecognized processor", 16, 8) },
  { label: "Unknown laptop — recognizable GPU, no matching laptop photo", identity: identity("Example vendor", "Unlisted laptop", "Portable"), value: hardware("Intel Core i7-13700H", 32, 20, [{name: "NVIDIA GeForce RTX 4060 Laptop GPU", memory: 8}], 14) },
  { label: "Unknown desktop — recognized discrete GPU", identity: identity("Example vendor", "Custom desktop", "Desktop"), value: hardware("Unrecognized processor", 64, 16, [{name: "NVIDIA GeForce RTX 5090", memory: 32}]) },
  { label: "Unknown desktop and unknown GPU", identity: identity("Example vendor", "Unlisted desktop", "Desktop"), value: hardware("Unrecognized processor", 32, 8, [{name: "Unrecognized accelerator", memory: 8}]) },
]
// Synthetic observations isolate presentation rules; these are not asserted retail configurations.
const detailExamples = [
  { label: "RTX 3050 Laptop · 4 GB — architecture and maximum bandwidth", note: "Shows ‘Up to 192 GB/s’. The GPU architecture label is now omitted.", identity: null, value: hardware("Intel Core i7-13700H", 32, 20, [{name: "NVIDIA GeForce RTX 3050 Laptop GPU", memory: 4}], 14) },
  { label: "RTX 3050 Laptop · 6 GB — same GPU name, different bandwidth", note: "The 6 GB variant selects ‘Up to 168 GB/s’; The GPU architecture label is omitted.", identity: null, value: hardware("Intel Core i7-13700H", 32, 20, [{name: "NVIDIA GeForce RTX 3050 Laptop GPU", memory: 6}], 14) },
  { label: "M4 — physical CPU count unavailable", note: "Omits the CPU count when it is unknown. The catalog’s ‘8 or 10’ range is already excluded from the card.", identity: identity("Apple", "Mac16,10"), value: hardware("Apple M4", 24, 10, [{name: "Apple M4", memory: 24, shared: true}]) },
  { label: "M4 — physical CPU count available", note: "Compare with the previous card: an observed 10-core CPU adds a definite count.", identity: identity("Apple", "Mac16,10"), value: hardware("Apple M4", 24, 10, [{name: "Apple M4", memory: 24, shared: true}], 10) },
  { label: "M4 Max — your chip configuration", note: "16 CPU cores, 40 GPU cores, and 546 GB/s bandwidth. Neural Engine cores are omitted.", identity: identity("Apple", "Mac16,5"), value: hardware("Apple M4 Max", 64, 16, [{name: "Apple M4 Max", memory: 64, shared: true}], 16) },
  { label: "Shared GPU memory — Strix Halo", note: "Keeps ‘Shares unified memory’, making clear that GPU memory is not an additional VRAM pool.", identity: identity("HP", "HP Z2 Mini G1a Workstation Desktop PC", "MiniPc"), value: hardware("AMD Ryzen AI Max+ PRO 395", 128, 32, [{name: "AMD Radeon 8060S Graphics", memory: 128, shared: true}], 16) },
  { label: "RTX 3060 · 12 GB — dedicated VRAM", note: "System RAM and GPU VRAM remain separate. CUDA core counts are omitted.", identity: null, value: hardware("AMD Ryzen 9 9950X", 64, 32, [{name: "NVIDIA GeForce RTX 3060", memory: 12}], 16) },
  { label: "RTX 3060 · 8 GB — memory-dependent specification", note: "Compare its bandwidth with the 12 GB version; the card uses the observed VRAM capacity to choose the specification.", identity: null, value: hardware("AMD Ryzen 9 9950X", 64, 32, [{name: "NVIDIA GeForce RTX 3060", memory: 8}], 16) },
]
const catalog = hardwarePhotoInventory.map(entry => {
  const row = coverage.find(row => row.photoId === entry.id)!
  const match = entry.match
  const names = match._tag === "Gpu" ? match.names : match.models
  return {entry, row, names}
})
createRoot(document.getElementById("root")!).render(<main className="mx-auto max-w-5xl p-6 text-slate-900 dark:text-slate-100">
  <style>{`html, body, #root {height:auto;min-height:100%} html {overflow-y:auto} body { overflow:visible; margin:0; background:${dark ? '#090f1b' : '#f8fafc'} } html {scroll-behavior:smooth} article {margin:28px 0 44px} main > h1 {font-size:30px;font-weight:700} main > section > h2 {font-size:23px;margin:36px 0 16px;font-weight:600} article > h3, .fact > h3 {font-size:17px;margin-bottom:12px;font-weight:600} main > p, main > section > p, details p, .fact > p {margin:10px 0;line-height:1.6} nav {display:flex;gap:18px;flex-wrap:wrap;margin:20px 0} a {text-decoration:underline} details {margin:12px 0;padding:12px;border:1px solid #64748b55;border-radius:10px} summary {cursor:pointer} .aliases {overflow-wrap:anywhere;color:${dark ? '#cbd5e1' : '#475569'}} .facts {display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px} .fact {padding:16px;border:1px solid #64748b55;border-radius:12px} :target {scroll-margin-top:20px}`}</style>
  <h1>Hardware gallery</h1>
  <p>All {catalog.length} photo groups, all matching identifiers, {facts.length} published specification entries, and unknown-hardware fallbacks.</p>
  <p>The hardware cards below use example configurations, not a scan of your computer. The photo catalog shows every image without inventing a configuration. Open each mapping to see all supported identifiers and researched configurations.</p>
  <nav><a href="#detail-examples">Details to review</a><a href="#unknown">Unknown hardware</a><a href="#scenarios">Configuration examples</a><a href="#catalog">All 44 photo options</a><a href="#facts">All specifications</a><a href={dark ? "?" : "?dark"}>{dark ? "Light mode" : "Dark mode"}</a></nav>
  <section id="detail-examples"><h2>Details to review · {detailExamples.length} examples</h2><p>Real hardware-card rendering with synthetic observations to expose each display rule. These examples are not scans of your machine or a list of verified retail configurations.</p>{detailExamples.map(s => <article key={s.label}><h3>{s.label}</h3><p>{s.note}</p><HardwareSummary identity={s.identity} value={s.value}/></article>)}</section>
  <section id="unknown"><h2>Unknown hardware</h2>{unknowns.map(s => <article key={s.label}><h3>{s.label}</h3><HardwareSummary identity={s.identity} value={s.value}/></article>)}</section>
  <section id="scenarios"><h2>Configuration examples</h2><p>Representative laptop, unified-memory, multi-GPU and AI mini PC configurations.</p>{hardwareScenarios.map(s => <article key={s.label}><h3>{s.label}</h3><HardwareSummary identity={s.identity} value={s.value}/></article>)}</section>
  <section id="catalog"><h2>All photo options</h2>{catalog.map(({entry,row,names},i) => <article id={entry.id} key={entry.id} data-photo-id={entry.id}>
    <h3>{i+1}. {entry.subject} · {entry.category}</h3><div className="mx-auto w-full max-w-[400px]"><HardwarePhotograph photo={entry}/></div>
    <details><summary>All matching identifiers and configuration coverage ({names.length} identifiers)</summary><p className="aliases">{names.join(" · ")}</p><p>Processors: {row.processors.join(", ") || "No CPU implied by a graphics-card photo."}</p><p>Accelerators: {row.accelerators.join(", ") || "Configuration-dependent."}</p><p>{row.configurationNote}</p>{Option.isSome(entry.processorModels) && <p>Required processor identifiers: {entry.processorModels.value.join(", ")}</p>}</details>
  </article>)}</section>
  <section id="facts"><h2>All published specifications</h2><p>Nominal facts are distinct from detected hardware. Configurable variants require sufficient observed identity.</p><div className="facts">{facts.map(f => <div className="fact" key={`${f.target}-${f.names[0]}`}><h3>{f.names[0]}</h3><p>{f.target}</p>{f.facts.map(v => <p key={v.label}>{v.label.replace(/ \(spec\)/g, "")}: <strong>{v.value}</strong></p>)}{"variants" in f && f.variants?.map(v => <p key={v.physicalCpuCores}>With {v.physicalCpuCores} observed physical CPU cores: {v.facts.map(x => `${x.label.replace(/ \(spec\)/g, "")}: ${x.value}`).join("; ")}</p>)}{"memoryVariants" in f && f.memoryVariants?.map(v => <p key={v.memoryGiB}>With {v.memoryGiB} GB VRAM: {v.facts.map(x => `${x.label.replace(/ \(spec\)/g, "")}: ${x.value}`).join("; ")}</p>)}<details><summary>Aliases and source</summary><p>{f.names.join(" · ")}</p><a href={f.source} target="_blank" rel="noreferrer">Primary source</a>{"additionalSources" in f && f.additionalSources?.map((url, i) => <p key={url}><a href={url} target="_blank" rel="noreferrer">Supporting source {i + 1}</a></p>)}</details></div>)}</div></section>
  <p><a href="#">Back to top</a></p>
</main>)
