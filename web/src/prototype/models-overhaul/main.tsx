import "@/styles/tailwind.css"
import { createRoot } from "react-dom/client"
import { initializeAppearance } from "@/stores/appearance-store"
import { Prototype } from "./prototype"

initializeAppearance("system")
createRoot(document.getElementById("root")!).render(<Prototype />)
