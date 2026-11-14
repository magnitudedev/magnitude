//! The registered model families (integration spec §4.3), listed once.
//!
//! The engine's recognition (`magnitude_engine::families`) runs over this
//! list, and so do the executor's catalog assessment and the catalog
//! definitions' pins. It lives below both the engine and the executor
//! because the engine depends on the executor. A new family is registered
//! here and nowhere else.

use magnitude_family_contracts::ModelFamily;
use magnitude_family_deepseek41::DeepSeek41Family;
use magnitude_family_gemma4::Gemma4Family;
use magnitude_family_glm5next::Glm5NextFamily;
use magnitude_family_lfm2::Lfm2Family;
use magnitude_family_llama::LlamaFamily;
use magnitude_family_muse_glimmer::MuseGlimmerFamily;
use magnitude_family_nemotron_h::NemotronHFamily;
use magnitude_family_qwen35::Qwen35Family;
use magnitude_family_qwen4exp::Qwen4ExpFamily;

/// Every target family. Each architecture is claimed by exactly one of them
/// (`magnitude_engine::families::recognize` refuses any other outcome).
pub static FAMILIES: &[&dyn ModelFamily] = &[
    &Qwen35Family,
    &Qwen4ExpFamily,
    &LlamaFamily,
    &NemotronHFamily,
    &Lfm2Family,
    &Gemma4Family,
    &MuseGlimmerFamily,
    &DeepSeek41Family,
    &Glm5NextFamily,
];
