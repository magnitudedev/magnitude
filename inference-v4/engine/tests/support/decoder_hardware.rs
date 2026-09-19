//! Functional decoder machines only. Every cost is an explicit hypothetical
//! instruction service; no measurement or device-name inference supplies it.
use seismic_accounting::{
    schedule::{CapacityUnit, Resource, Timebase},
    workload::DerivationLimits,
};
use seismic_runtime::{
    plan::Settings,
    tuner::{Form, Hardware},
};

fn settings(hardware: Hardware, form: Form) -> Settings {
    Settings {
        hardware,
        form,
        derivation_limits: DerivationLimits {
            instructions: 10_000_000,
            operations: 1_000_000,
        },
        search: seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 1_000_000, ..Default::default() }, ..Default::default() },
    }
}

/// Inventory actual typed computations without executing or selecting native
/// candidates. These signatures seed a reusable functional scalar fixture.
fn scalar_computations() -> Vec<seismic_lang::lowered_ir::LoweredIr> {
    use seismic_lang::{
        lower::{lower_specialized, Options},
        types::{DType, Elem},
    };
    use std::collections::HashMap;
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    let cases: &[(&str, &[(&str, i64)])] = &[
        ("qwen_embedding_rows", &[("M", 1), ("V", 32), ("D", 8)]),
        ("qwen_readout_rows", &[("M", 1), ("V", 32), ("D", 8)]),
        ("qwen_dense_suffix", &[("M", 1), ("H", 8), ("F", 12)]),
        (
            "qwen_recurrent_sequence",
            &[("M", 1), ("H", 8), ("NK", 2), ("GV", 2), ("W", 4), ("C", 4)],
        ),
        (
            "qwen_attention_sequence",
            &[
                ("M", 1),
                ("D", 8),
                ("T", 16),
                ("R", 1),
                ("H", 4),
                ("KV", 2),
                ("P", 6),
                ("S", 4),
                ("SH", 2),
                ("SW", 1),
            ],
        ),
        (
            "qwen_routed_suffix",
            &[("M", 1), ("H", 8), ("E", 7), ("K", 3), ("F", 12), ("S", 16)],
        ),
    ];
    cases
        .iter()
        .map(|(name, dimensions)| {
            let source = program.functions.iter().find(|f| f.name == *name).unwrap();
            let elements = source
                .elem_params
                .iter()
                .map(|name| {
                    (
                        name.clone(),
                        Elem::Dtype(if name == "SRW" {
                            DType::F32
                        } else {
                            DType::BF16
                        }),
                    )
                })
                .collect();
            let shapes: HashMap<_, _> = dimensions
                .iter()
                .map(|(n, v)| (n.to_string(), *v))
                .collect();
            lower_specialized(
                &program,
                name,
                "cpu",
                &shapes,
                &elements,
                &Options::default(),
            )
            .unwrap()
        })
        .collect()
}

pub fn cpu() -> Settings {
    use seismic_accounting::execution_model::{
        self, PrimitiveKind, PrimitiveTiming, ScalarHardware, Scope,
    };
    use seismic_accounting::schedule::Reservation;
    let mut patterns = Vec::new();
    for lowered in scalar_computations() {
        for loads in [
            seismic_realization::LoadStrategy::Materialize,
            seismic_realization::LoadStrategy::BorrowProvenReadOnly,
        ] {
            let prepared = seismic_cpu::prepare(&lowered, loads).unwrap();
            for primitive in execution_model::requirements(&prepared).unwrap() {
                // A source math helper is not a hardware instruction mapping.
                if matches!(primitive.kind, PrimitiveKind::Math(_)) {
                    continue;
                }
                let pattern = primitive.signature();
                if !patterns.contains(&pattern) {
                    patterns.push(pattern);
                }
            }
        }
    }
    settings(
        Hardware::Cpu(ScalarHardware {
            identity: "hypothetical decoder scalar service; opaque helpers unresolved".into(),
            scope: Scope::HypotheticalDirectScalarV1,
            timebase: Timebase {
                seconds_numerator: 1,
                seconds_denominator: 1,
            },
            resources: vec![Resource {
                name: "instruction service".into(),
                capacity: 1,
                unit: CapacityUnit::Slots,
            }],
            timings: patterns
                .into_iter()
                .map(|primitive| PrimitiveTiming {
                    primitive,
                    latency: 1,
                    services: vec![Reservation {
                        resource: 0,
                        offset: 0,
                        duration: 1,
                        units: 1,
                    }],
                })
                .collect(),
        }),
        Form::CpuScalar,
    )
}

#[cfg(target_os = "macos")]
pub fn metal() -> Settings {
    use seismic_lang::{
        ast::{BinaryOp, UnaryOp},
        types::DType,
    };
    use seismic_metal::{
        collective::FragmentLayout,
        model,
        terminal::{Primitive, Space, Type},
    };
    let types = [
        Type::Bool,
        Type::I32,
        Type::U32,
        Type::I64,
        Type::U64,
        Type::F16,
        Type::BF16,
        Type::F32,
    ];
    let spaces = [
        Space::Private,
        Space::Threadgroup,
        Space::Device,
        Space::Constant,
    ];
    let mut primitives = vec![
        Primitive::Branch,
        Primitive::Return,
        Primitive::Select,
        Primitive::Barrier,
        Primitive::Launch,
        Primitive::Group,
    ];
    // Cover the backend's typed primitive vocabulary, independently of which
    // decomposition, placement, vector cover, or matrix body selection chooses.
    for ty in types {
        for operation in [
            BinaryOp::Or,
            BinaryOp::And,
            BinaryOp::Eq,
            BinaryOp::Ne,
            BinaryOp::Lt,
            BinaryOp::Le,
            BinaryOp::Gt,
            BinaryOp::Ge,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::BitAnd,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Rem,
        ] {
            primitives.push(Primitive::Binary { operation, ty });
        }
        for operation in [UnaryOp::Neg, UnaryOp::Not, UnaryOp::BitNot] {
            primitives.push(Primitive::Unary { operation, ty });
        }
        for to in types {
            primitives.push(Primitive::Cast { from: ty, to });
            primitives.push(Primitive::Bitcast { from: ty, to });
        }
        for space in spaces {
            primitives.extend([
                Primitive::Read { space, ty },
                Primitive::Write { space, ty },
                Primitive::Address { space, ty },
            ]);
            for components in [2, 3, 4] {
                primitives.push(Primitive::VectorRead {
                    space,
                    ty,
                    components,
                });
            }
        }
        for name in [
            "exp",
            "fast::exp",
            "rsqrt",
            "sqrt",
            "log",
            "sin",
            "cos",
            "abs",
            "simd_sum",
            "simd_max",
            "simd_min",
        ] {
            primitives.push(Primitive::Builtin {
                name: name.into(),
                inputs: vec![ty],
                result: ty,
            });
        }
        for name in ["max", "min"] {
            primitives.push(Primitive::Builtin {
                name: name.into(),
                inputs: vec![ty, ty],
                result: ty,
            });
        }
        primitives.push(Primitive::Builtin {
            name: "fma".into(),
            inputs: vec![ty; 3],
            result: ty,
        });
        primitives.push(Primitive::Builtin {
            name: "simd_shuffle".into(),
            inputs: vec![ty, Type::U32],
            result: ty,
        });
    }
    let layouts =
        [DType::F16, DType::BF16, DType::F32].map(|dtype| FragmentLayout::metal(dtype).unwrap());
    for layout in layouts {
        for space in [Space::Device, Space::Threadgroup] {
            primitives.push(Primitive::MatrixStore { layout, space });
            for transpose in [false, true] {
                primitives.push(Primitive::MatrixLoad {
                    layout,
                    space,
                    transpose,
                });
            }
        }
        for right in layouts {
            for accumulator in layouts {
                for output in layouts {
                    primitives.push(Primitive::MatrixMultiplyAccumulate {
                        layouts: [layout, right, accumulator, output],
                    });
                }
            }
        }
    }
    let hardware = model::Hardware {
        identity: "hypothetical single-service decoder machine; functional numerical fixture only"
            .into(),
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![Resource {
            name: "terminal operation service".into(),
            capacity: 1,
            unit: CapacityUnit::Slots,
        }],
        resident_groups: 1,
        resident_shared_bytes: 65536,
        timings: primitives
            .into_iter()
            .map(|primitive| model::Timing {
                primitive,
                latency: 1,
                services: vec![model::Service {
                    resource: 0,
                    offset: 0,
                    duration: 1,
                    units: model::Units::PerSubgroup(1),
                }],
            })
            .collect(),
    };
    hardware.validate().unwrap();
    settings(Hardware::Metal(hardware), Form::Metal)
}
