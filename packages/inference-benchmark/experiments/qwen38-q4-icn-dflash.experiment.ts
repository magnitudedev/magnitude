import { qwen38 } from "../models/qwen3.8-27b.model"
import { defineIcnDflashComparison } from "./icn-dflash-comparison"

export default defineIcnDflashComparison({
  id: "qwen38-q4-icn-dflash",
  title: "Qwen3.8 27B Q4: ICN standalone vs ICN DFlash2 across context depths",
  contexts: [4_096, 16_384, 32_768, 65_536],
  target: qwen38.artifacts.llamaQ4,
  drafter: qwen38.artifacts.dflashQ8,
})
