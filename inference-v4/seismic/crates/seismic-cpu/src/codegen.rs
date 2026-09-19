//! Retained native compilation policy, resolved before native compilation.
//!
//! Host feature discovery supplies target input. It does not compile a candidate.
//! This pins external compiler settings and import ABI; it does not assert that
//! Cranelift instruction selection or register allocation preserves model costs.
use cranelift_codegen::{
    ir::{self, types, InstructionData},
    isa,
    settings::{self, Configurable},
};
use seismic_realization::{MathFunction, ScalarProgram};
use std::collections::{BTreeMap, BTreeSet};

const CODEGEN_VERSION: &str = "cranelift-codegen/0.125.4";
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Policy {
    compiler: String,
    triple: String,
    call_conv: isa::CallConv,
    shared: BTreeMap<String, String>,
    isa: BTreeMap<String, String>,
}
pub struct Target {
    pub(crate) isa: isa::OwnedTargetIsa,
    policy: Policy,
}
impl Policy {
    pub fn host() -> Result<Self, String> {
        let mut flags = settings::builder();
        for (name, value) in [
            ("use_colocated_libcalls", "false"),
            ("is_pic", "false"),
            ("opt_level", "speed"),
            ("machine_code_cfg_info", "true"),
        ] {
            flags.set(name, value).map_err(|e| e.to_string())?;
        }
        let isa = cranelift_native::builder()
            .map_err(str::to_owned)?
            .finish(settings::Flags::new(flags))
            .map_err(|e| e.to_string())?;
        Ok(Self::from_isa(isa.as_ref()))
    }
    fn from_isa(isa: &dyn isa::TargetIsa) -> Self {
        Self {
            compiler: CODEGEN_VERSION.into(),
            triple: isa.triple().to_string(),
            call_conv: isa.default_call_conv(),
            shared: isa
                .flags()
                .iter()
                .map(|f| (f.name.into(), f.value_string()))
                .collect(),
            isa: isa
                .isa_flags()
                .into_iter()
                .map(|f| (f.name.into(), f.value_string()))
                .collect(),
        }
    }
    pub fn compiler(&self) -> &str {
        &self.compiler
    }
    /// Calling convention of every compiled phase and math import.
    pub fn call_conv(&self) -> isa::CallConv {
        self.call_conv
    }
    pub fn triple(&self) -> &str {
        &self.triple
    }
    pub fn shared_flags(&self) -> &BTreeMap<String, String> {
        &self.shared
    }
    pub fn isa_flags(&self) -> &BTreeMap<String, String> {
        &self.isa
    }
    pub fn target(&self) -> Result<Target, String> {
        if self.compiler != CODEGEN_VERSION {
            return Err("CPU compiler version differs from retained policy".into());
        }
        // Current native entry is a host executable. An artifact cannot silently
        // migrate to a different feature set while retaining its old conditions.
        if *self != Self::host()? {
            return Err("CPU host target or flags differ from retained codegen policy".into());
        }
        let mut shared = settings::builder();
        for (name, value) in &self.shared {
            shared.set(name, value).map_err(|e| e.to_string())?;
        }
        let mut builder = isa::lookup(
            self.triple
                .parse()
                .map_err(|e| format!("invalid target triple: {e}"))?,
        )
        .map_err(|e| e.to_string())?;
        for (name, value) in &self.isa {
            builder.set(name, value).map_err(|e| e.to_string())?;
        }
        let isa = builder
            .finish(settings::Flags::new(shared))
            .map_err(|e| e.to_string())?;
        if Self::from_isa(isa.as_ref()) != *self {
            return Err("CPU target reconstruction changed retained settings".into());
        }
        Ok(Target {
            isa,
            policy: self.clone(),
        })
    }
}
impl Target {
    pub fn policy(&self) -> &Policy {
        &self.policy
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Import {
    pub reference: ir::FuncRef,
    pub operation: MathFunction,
    pub symbol: String,
    pub signature: ir::Signature,
    /// This is an ABI/semantic identity, not a hardware instruction contract.
    pub implementation: MathRuntime,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathRuntime {
    /// The Rust standard-library binary64 function, rounded once to binary32: what the
    /// reference interpreter computes.
    RustStandardLibraryF64Rounded,
}

pub fn imports(program: &ScalarProgram) -> Result<Vec<Import>, String> {
    let mut refs = BTreeSet::new();
    let mut imports = Vec::new();
    for &(reference, operation) in &program.imports {
        if !refs.insert(reference) {
            return Err("CPU math import reference is duplicated".into());
        }
        let external = program
            .function
            .dfg
            .ext_funcs
            .get(reference)
            .ok_or("CPU import reference is absent")?;
        let signature = program
            .function
            .dfg
            .signatures
            .get(external.signature)
            .ok_or("CPU import signature is absent")?
            .clone();
        let mut expected = ir::Signature::new(program.function.signature.call_conv);
        expected.params.push(ir::AbiParam::new(types::F32));
        expected.returns.push(ir::AbiParam::new(types::F32));
        if signature != expected || external.colocated {
            return Err("CPU math import has an incompatible ABI".into());
        }
        imports.push(Import {
            reference,
            operation,
            symbol: operation.symbol().into(),
            signature,
            implementation: MathRuntime::RustStandardLibraryF64Rounded,
        });
    }
    for block in program.function.layout.blocks() {
        for inst in program.function.layout.block_insts(block) {
            match program.function.dfg.insts[inst] {
                InstructionData::Call { func_ref, .. } if refs.contains(&func_ref) => {}
                _ if program.function.dfg.insts[inst].opcode().is_call() => {
                    return Err("CPU execution has an unbound native call".into())
                }
                _ => {}
            }
        }
    }
    Ok(imports)
}
