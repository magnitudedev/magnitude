import { qwen38 } from "../models/qwen3.8-27b.model"
import { defineIcnDflashSmoke } from "./icn-dflash-comparison"

export default defineIcnDflashSmoke({
  id: "qwen38-q4-icn-dflash-smoke",
  title: "Qwen3.8 27B Q4: ICN standalone vs ICN DFlash2 (smoke)",
  target: qwen38.artifacts.llamaQ4,
  drafter: qwen38.artifacts.dflashQ8,
})
