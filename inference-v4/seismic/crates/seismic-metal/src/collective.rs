//! Admission of Metal subgroup primitives before target printing. The returned
//! implementation is consumed by both emission and resource accounting.
use crate::memory::{ControlValue, Scope};
use seismic_lang::{
    exec::ir::*,
    exec::types::Ty,
    intrinsics::Operation,
    types::{DType, Elem},
};
use seismic_realization::{dispatch::TilePlacement, execution::Multiplicity};
use std::sync::Arc;

use std::collections::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FragmentLayout {
    pub rows: u64,
    pub columns: u64,
    pub dtype: DType,
}
impl FragmentLayout {
    pub fn metal(dtype: DType) -> Result<Self, String> {
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
            return Err(format!(
                "Metal SIMD-group matrices do not support {}",
                dtype.name()
            ));
        }
        Ok(Self {
            rows: 8,
            columns: 8,
            dtype,
        })
    }
    pub fn from_type(ty: &Ty) -> Result<Self, String> {
        let Ty::Frag(shape) = ty else {
            return Err("collective operand is not a fragment".into());
        };
        let Elem::Dtype(dtype) = shape.elem else {
            return Err("fragment element type is unresolved".into());
        };
        let layout = Self::metal(dtype)?;
        if shape.shape.len() != 2
            || shape.shape[0].as_constant() != Some(layout.rows as i64)
            || shape.shape[1].as_constant() != Some(layout.columns as i64)
        {
            return Err("unsupported SIMD-group fragment shape".into());
        }
        Ok(layout)
    }
    pub fn elements(self) -> u64 {
        self.rows * self.columns
    }
    pub fn payload_bytes(self) -> u64 {
        self.elements() * u64::from(self.dtype.bytes())
    }
    pub fn metal_type(self) -> Result<&'static str, String> {
        if self != Self::metal(self.dtype)? {
            return Err("fragment layout differs from admitted Metal matrix implementation".into());
        }
        Ok(match self.dtype {
            DType::F16 => "simdgroup_half8x8",
            DType::BF16 => "simdgroup_bfloat8x8",
            DType::F32 => "simdgroup_float8x8",
            _ => unreachable!(),
        })
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct FragmentAllocation {
    pub operation: OperationId,
    pub variable: VarId,
    pub layout: FragmentLayout,
    pub scope: Vec<Scope>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageSpace {
    Device,
    Threadgroup,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixMemory {
    /// The selected view expression, including shape/stride transformation. All
    /// address evaluation and validity operations belong to its implementation.
    pub operand: Expr,
    pub row: Expr,
    pub column: Expr,
    pub space: StorageSpace,
    pub dtype: DType,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Implementation {
    ParticipantIndex,
    Exchange {
        dtype: DType,
    },
    Reduction {
        operation: Operation,
        dtype: DType,
    },
    Declare {
        fragment: VarId,
        layout: FragmentLayout,
    },
    Load {
        fragment: VarId,
        layout: FragmentLayout,
        memory: MatrixMemory,
        transpose: bool,
    },
    Store {
        fragment: VarId,
        layout: FragmentLayout,
        memory: MatrixMemory,
    },
    MultiplyAccumulate {
        fragments: [VarId; 4],
        layouts: [FragmentLayout; 4],
    },
}
impl Implementation {
    pub fn operation(&self) -> Operation {
        match self {
            Self::ParticipantIndex => Operation::LaneIndex,
            Self::Exchange { .. } => Operation::ShuffleIndex,
            Self::Reduction { operation, .. } => *operation,
            Self::Declare { .. } => Operation::Matrix,
            Self::Load {
                transpose: false, ..
            } => Operation::MatrixLoad,
            Self::Load {
                transpose: true, ..
            } => Operation::MatrixLoadTranspose,
            Self::Store { .. } => Operation::MatrixStore,
            Self::MultiplyAccumulate { .. } => Operation::MatrixMultiplyAccumulate,
        }
    }
    /// Selected target builtin, consumed by emission and resource-mapping keys.
    pub fn metal_builtin(&self) -> Option<&'static str> {
        match self {
            Self::ParticipantIndex => None,
            Self::Exchange { .. } => Some("simd_shuffle"),
            Self::Declare { .. } => None,
            Self::Load { .. } => Some("simdgroup_load"),
            Self::Store { .. } => Some("simdgroup_store"),
            Self::MultiplyAccumulate { .. } => Some("simdgroup_multiply_accumulate"),
            Self::Reduction { operation, .. } => Some(operation.name()),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::ParticipantIndex => {}
            Self::Exchange { dtype } => {
                if !dtype.is_float() {
                    return Err("Metal exchange requires floating scalar storage".into());
                }
            }
            Self::Reduction { operation, dtype } => {
                if !matches!(
                    operation,
                    Operation::SimdSum | Operation::SimdMax | Operation::SimdMin
                ) || !dtype.is_float()
                {
                    return Err("invalid Metal scalar collective contract".into());
                }
            }
            Self::Declare { layout, .. } => {
                layout.metal_type()?;
            }
            Self::Load { layout, memory, .. } | Self::Store { layout, memory, .. } => {
                layout.metal_type()?;
                if memory.dtype != layout.dtype {
                    return Err(
                        "matrix transfer element type must match fragment storage type".into(),
                    );
                }
                if matches!(self, Self::Store { .. }) && memory.space != StorageSpace::Threadgroup {
                    return Err("selected matrix store requires threadgroup tile storage".into());
                }
            }
            Self::MultiplyAccumulate { layouts, .. } => {
                for layout in layouts {
                    layout.metal_type()?;
                }
                // Each operand is consumed at its own float type and widened exactly to the
                // accumulator's: bfloat x float atoms accumulating in float measured bit-identical
                // to the F32 FMA chain (Apple M4 Max, 2026-09-19). The accumulator and the
                // result share one type, and no operand is wider than it.
                let wider = |operand: DType| operand.bytes() > layouts[0].dtype.bytes();
                if layouts[0].dtype != layouts[3].dtype || !layouts[1].dtype.is_float() || !layouts[2].dtype.is_float() || wider(layouts[1].dtype) || wider(layouts[2].dtype) {
                    return Err(
                        "matrix multiply accumulator and result types must agree, and no float operand may be wider than them"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Site {
    pub operation: OperationId,
    pub ordinal: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Collective {
    pub site: Site,
    pub implementation: Implementation,
    pub scope: Vec<Scope>,
    /// One invocation is one full SIMD-group operation, never one invocation per
    /// lane. Dynamic control remains explicit and does not imply convergence.
    pub executions: Arc<Multiplicity<ControlValue>>,
}

pub(crate) fn implementation(
    op: Operation,
    args: &[Expr],
    output: Option<VarId>,
    bound: &HashMap<VarId, Option<TilePlacement>>,
) -> Result<Implementation, String> {
    let variable = |expr: &Expr| match expr.kind {
        ExprKind::Var(v) => Ok(v),
        _ => Err("collective fragment must be a bound variable".to_string()),
    };
    let layout = |expr: &Expr| FragmentLayout::from_type(&expr.ty);
    let instruction = match op {
        Operation::LaneIndex => {
            if !args.is_empty() {
                return Err("lane index has no operands".into());
            }
            Implementation::ParticipantIndex
        }
        Operation::ShuffleIndex => {
            let [value, index] = args else {
                return Err("shuffle needs a value and participant index".into());
            };
            let Ty::Scalar(dtype) = value.ty else {
                return Err("shuffle needs a scalar value".into());
            };
            if !matches!(index.ty,Ty::Scalar(d) if d.is_int()) {
                return Err("shuffle participant index is not integral".into());
            }
            Implementation::Exchange { dtype }
        }
        Operation::Matrix => {
            let Ty::Scalar(dtype) = args.first().ok_or("matrix dtype missing")?.ty else {
                return Err("matrix dtype is not a scalar type".into());
            };
            Implementation::Declare {
                fragment: output.ok_or("matrix declaration requires a direct value binding")?,
                layout: FragmentLayout::metal(dtype)?,
            }
        }
        Operation::SimdSum | Operation::SimdMax | Operation::SimdMin => {
            let Ty::Scalar(dtype) = args.first().ok_or("collective input missing")?.ty else {
                return Err("SIMD reduction needs a scalar input".into());
            };
            Implementation::Reduction {
                operation: op,
                dtype,
            }
        }
        Operation::MatrixLoad | Operation::MatrixLoadTranspose | Operation::MatrixStore => {
            if args.len() != 4 {
                return Err("matrix memory primitive requires four operands".into());
            }
            let fragment = variable(&args[0])?;
            let layout = layout(&args[0])?;
            let tile = args[1]
                .ty
                .shaped()
                .ok_or("matrix memory operand is unshaped")?;
            let Elem::Dtype(dtype) = tile.elem else {
                return Err("matrix transfer requires dense elements".into());
            };
            let root = crate::storage::tile_root(&args[1])
                .ok_or("matrix memory operand has no allocation root")?;
            let space = match bound.get(&root) {
                Some(Some(TilePlacement::GroupShared | TilePlacement::GroupWide)) => StorageSpace::Threadgroup,
                Some(None) => StorageSpace::Device,
                _ => return Err("matrix operand lacks shared or borrowed device storage".into()),
            };
            let memory = MatrixMemory {
                operand: args[1].clone(),
                row: args[2].clone(),
                column: args[3].clone(),
                space,
                dtype,
            };
            if op == Operation::MatrixStore {
                Implementation::Store {
                    fragment,
                    layout,
                    memory,
                }
            } else {
                Implementation::Load {
                    fragment,
                    layout,
                    memory,
                    transpose: op == Operation::MatrixLoadTranspose,
                }
            }
        }
        Operation::MatrixMultiplyAccumulate => {
            if args.len() != 4 {
                return Err("matrix multiply needs four operands".into());
            }
            Implementation::MultiplyAccumulate {
                fragments: [
                    variable(&args[0])?,
                    variable(&args[1])?,
                    variable(&args[2])?,
                    variable(&args[3])?,
                ],
                layouts: [
                    layout(&args[0])?,
                    layout(&args[1])?,
                    layout(&args[2])?,
                    layout(&args[3])?,
                ],
            }
        }
    };
    instruction.validate()?;
    Ok(instruction)
}
