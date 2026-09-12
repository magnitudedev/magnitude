import pi from "../../assets/brand/harnesses/pi.svg"
import opencode from "../../assets/brand/harnesses/opencode.svg"
import hermes from "../../assets/brand/harnesses/hermes.svg"
import openclaw from "../../assets/brand/harnesses/openclaw.svg"
import codex from "../../assets/brand/harnesses/codex.svg"
import claude from "../../assets/brand/harnesses/claude.svg"
import omp from "../../assets/brand/harnesses/omp.svg"
import cline from "../../assets/brand/harnesses/cline.svg"
import type { HarnessId } from "@magnitudedev/client-common"
const logos: Record<string,string> = {pi,opencode,hermes,openclaw,codex,"claude-code":claude,"oh-my-pi":omp,cline}
export function HarnessLogo({ id, name }: { id: HarnessId; name: string }) {
 return <span className="flex size-14 shrink-0 items-center justify-center rounded-2xl border border-slate-700 bg-slate-900 p-3"><img src={logos[id]} alt={`${name} logo`} className={`size-full object-contain ${id === "cline" ? "invert" : ""}`} /></span>
}
