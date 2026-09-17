use cranelift_codegen::isa::CallConv;
use seismic_lang::program::{collect_files, compile};
use seismic_realization::Dispatch;
use std::{collections::HashMap, path::PathBuf};

#[test]
fn portable_projection_and_norm_emit_direct_ptx() {
    let p = compile(
        &collect_files(&[
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../seismic-std/lib")
        ])
        .unwrap(),
        &["cuda".into(), "metal".into(), "cpu".into()],
    )
    .unwrap_or_else(|e| panic!("{e:?}"));
    for (name, shapes) in [
        (
            "projection",
            HashMap::from([("N".into(), 7), ("K".into(), 64)]),
        ),
        (
            "rms_norm",
            HashMap::from([("R".into(), 3), ("W".into(), 17)]),
        ),
    ] {
        let function = p.functions.iter().find(|f| f.name == name).unwrap();
        let elements = function
            .elem_params
            .iter()
            .map(|p| {
                (
                    p.clone(),
                    seismic_lang::types::Elem::Dtype(seismic_lang::types::DType::BF16),
                )
            })
            .collect();
        let lowered = seismic_lang::lower::lower_specialized(
            &p,
            name,
            "cuda",
            &shapes,
            &elements,
            &Default::default(),
        )
        .unwrap();
        let program =
            seismic_compiler::scalar_with(&lowered, CallConv::SystemV, Dispatch::ParallelRoot)
                .unwrap();
        let ptx = seismic_cuda::ptx::emit(&program).unwrap();
        assert!(ptx.contains(".visible .entry seismic_kernel"));
        assert!(ptx.contains("st.global.u32 [%status_addr]"));
        if let Ok(dir) = std::env::var("SEISMIC_PTX_OUTPUT") {
            std::fs::write(PathBuf::from(dir).join(format!("{name}.ptx")), ptx).unwrap();
        }
    }
}
