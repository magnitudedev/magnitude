//! Invocation validation (package R1).
//!
//! `CompiledPlan::prepare` is the only public constructor of
//! `PreparedInvocation`. It validates complete buffer/scalar/shape binding
//! sets, device ownership, type, actual byte requirements, alignment,
//! aliases, exact/bounded shape domains, and every invocation predicate;
//! evaluates every invocation expression once into dense validated words;
//! and allocates the invocation's result planes before submission. Submission
//! and backend execution accept `PreparedInvocation`, never raw bindings.

use crate::plan::{CompiledArtifact, ResultPlane};
#[cfg(target_os = "macos")]
use crate::plan::SealedBackend;
use crate::Device;
use seismic_lang::logical::specialization::ShapeFieldId;
use seismic_lang::types::DType;
use seismic_realization::failure::{
    ExecutionFailure, ExternalFailure, ExternalStage, InvalidInvocation,
};
use seismic_realization::ids::{BufferSlot, NativeFactIx, ScalarSlot};
use seismic_realization::invocation::{
    AliasRule, BufferRole, InvocationValues, ScalarContract, ScalarWord,
    ShapeFieldContract,
};
use std::sync::Arc;

/// Raw caller bindings by name; validated exactly once by `prepare`.
pub trait Bindings {
    /// One bound root-ABI tensor parameter, by interface parameter name and
    /// representation plane (`""` denotes the dense plane).
    fn buffer(&self, root: &str, plane: &str) -> Option<&crate::Buffer>;
    /// One bound by-value scalar, by contract scalar name.
    fn scalar(&self, name: &str) -> Option<f64>;
    /// The actual value of one bounded shape parameter, by field name.
    fn shape(&self, name: &str) -> Option<u64>;
}

/// One validated buffer: device-owned, sized, aligned, alias-checked.
#[derive(Clone)]
pub struct ValidatedBuffer {
    pub buffer: crate::Buffer,
    /// The actual validated byte requirement of this slot's contract entry.
    pub bytes: u64,
}

/// A validated invocation paired with its sealed artifact.
pub struct PreparedInvocation {
    pub(crate) artifact: Arc<CompiledArtifact>,
    /// Validated buffers in contract slot order (dense by `BufferSlot`).
    pub(crate) buffers: Vec<ValidatedBuffer>,
    pub(crate) values: InvocationValues,
    /// Owned result planes allocated by preparation, in ABI result order.
    pub(crate) results: Vec<ResultPlane>,
    /// The Metal handles of the result planes (a Metal artifact's executor
    /// binds them; they share the result planes' allocations).
    #[cfg(target_os = "macos")]
    pub(crate) metal_result_planes: Vec<seismic_metal::runtime::OwnedResultPlane>,
}

impl PreparedInvocation {
    pub fn artifact(&self) -> &CompiledArtifact {
        &self.artifact
    }

    /// Validated buffers dense by `BufferSlot` (contract slot order).
    pub fn buffers(&self) -> &[ValidatedBuffer] {
        &self.buffers
    }

    /// One validated buffer by contract slot; in-bounds by construction.
    pub fn buffer(&self, slot: BufferSlot) -> &ValidatedBuffer {
        &self.buffers[slot.index()]
    }

    /// The invocation values evaluated exactly once by `prepare`.
    pub fn values(&self) -> &InvocationValues {
        &self.values
    }

    /// The owned result planes, in ABI result order.
    pub fn results(&self) -> &[ResultPlane] {
        &self.results
    }
}

/// The R1-side of `prepare`: evaluate the contract once, validate every bound
/// buffer, and allocate the result planes. Invocation facts are
/// `ExecutionFailure::Invocation`; the only other failure is the external
/// allocation system.
pub(crate) fn prepare(
    device: &Device,
    artifact: Arc<CompiledArtifact>,
    parameters: &[String],
    bindings: &dyn Bindings,
) -> Result<PreparedInvocation, ExecutionFailure> {
    let contract = artifact.contract();
    // The parameter-name list and the contract were sealed together from one
    // pipeline result, and the seal emits one ABI buffer ordinal per interface
    // parameter: every contract ordinal indexes the list by construction.
    #[cfg(target_os = "macos")]
    let metal_backend = matches!(artifact.backend(), SealedBackend::Metal { .. });
    let values = contract.evaluate(
        &mut |entry: &ScalarContract| match bindings.scalar(&entry.name) {
            None => Err(InvalidInvocation::MissingScalar {
                slot: entry.slot,
                name: entry.name.clone(),
            }),
            Some(value) => encode_scalar(entry.slot, entry.dtype, value),
        },
        &mut |field: ShapeFieldId, entry: &ShapeFieldContract| {
            bindings.shape(&entry.name).ok_or(InvalidInvocation::MissingShape {
                field,
                name: entry.name.clone(),
            })
        },
        &|index: NativeFactIx| artifact.native_fact(index),
    )?;

    let mut buffers: Vec<ValidatedBuffer> = Vec::with_capacity(contract.buffers().len());
    let mut results: Vec<ResultPlane> = Vec::new();
    #[cfg(target_os = "macos")]
    let mut metal_result_planes: Vec<seismic_metal::runtime::OwnedResultPlane> =
        Vec::new();
    for entry in contract.buffers() {
        // The byte requirement is an expression of the actual validated
        // extents, already evaluated above.
        let required = values.derived[entry.bytes];
        let buffer = match entry.role {
            BufferRole::Parameter { ordinal, .. } => {
                let name = &parameters[ordinal as usize];
                let plane = plane_binding_name(&entry.plane);
                match bindings.buffer(name, plane) {
                    None => {
                        return Err(InvalidInvocation::MissingBuffer {
                            slot: entry.slot,
                            path: path_string(&entry.path),
                            plane: entry.plane.clone(),
                        }
                        .into())
                    }
                    Some(buffer) => buffer.clone(),
                }
            }
            BufferRole::Result => {
                let bytes = usize::try_from(required).map_err(|_| allocation_detail(
                    "a result plane exceeds the host address range",
                ))?;
                device.buffer(bytes).map_err(allocation_failure)?
            }
        };
        if !buffer.belongs_to(device) {
            return Err(InvalidInvocation::BufferDevice { slot: entry.slot }.into());
        }
        let actual = buffer.len() as u64;
        if actual < required {
            return Err(InvalidInvocation::BufferBytes {
                slot: entry.slot,
                required,
                actual,
            }
            .into());
        }
        let offset = buffer.allocation_offset();
        let actual_alignment = if offset == 0 {
            u64::MAX
        } else {
            1u64 << (offset as u64).trailing_zeros()
        };
        if actual_alignment < entry.alignment {
            return Err(InvalidInvocation::BufferAlignment {
                slot: entry.slot,
                required: entry.alignment,
                actual: actual_alignment,
            }
            .into());
        }
        if matches!(entry.role, BufferRole::Result) {
            #[cfg(target_os = "macos")]
            if metal_backend {
                // The plane was allocated by this Metal device and charged to
                // its domain, so its storage is the Metal arm; a different
                // storage kind would be a wrong-device buffer.
                let metal = match buffer.as_metal() {
                    Some(metal) => metal.clone(),
                    None => {
                        return Err(InvalidInvocation::BufferDevice { slot: entry.slot }
                            .into())
                    }
                };
                metal_result_planes.push(seismic_metal::runtime::OwnedResultPlane {
                    path: entry.path.0.clone(),
                    plane: entry.plane.clone(),
                    buffer: metal,
                });
            }
            results.push(ResultPlane {
                path: entry.path.0.clone(),
                plane: entry.plane.clone(),
                buffer: buffer.clone(),
            });
        }
        buffers.push(ValidatedBuffer {
            buffer,
            bytes: required,
        });
    }

    // Alias rules over actual byte ranges of the validated bindings:
    // shared-parameter pairs may overlap, every other pair must be disjoint.
    for alias in contract.aliases() {
        if alias.rule == AliasRule::MustDisjoint {
            let (left, right) = (
                &buffers[alias.left.index()],
                &buffers[alias.right.index()],
            );
            let (left_offset, right_offset) = (
                left.buffer.allocation_offset(),
                right.buffer.allocation_offset(),
            );
            let (left_end, right_end) = (
                left_offset.saturating_add(saturate(left.bytes)),
                right_offset.saturating_add(saturate(right.bytes)),
            );
            let overlapping = left.buffer.shares_allocation(&right.buffer)
                && left_offset < right_end
                && right_offset < left_end;
            if overlapping {
                return Err(InvalidInvocation::Alias {
                    left: alias.left,
                    right: alias.right,
                }
                .into());
            }
        }
    }

    Ok(PreparedInvocation {
        artifact,
        buffers,
        values,
        results,
        #[cfg(target_os = "macos")]
        metal_result_planes,
    })
}

/// The dense plane's source-level binding name.
fn plane_binding_name(plane: &str) -> &str {
    match plane {
        "dense" => "",
        other => other,
    }
}

fn path_string(path: &seismic_lang::types::ValuePath) -> String {
    path.0
        .iter()
        .map(|ordinal| ordinal.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

/// A byte count as the host's address width, saturating at its maximum.
fn saturate(bytes: u64) -> usize {
    match usize::try_from(bytes) {
        Ok(bytes) => bytes,
        Err(_) => usize::MAX,
    }
}

/// A result-plane allocation failure as the external allocation stage.
fn allocation_detail(reason: &str) -> ExecutionFailure {
    ExecutionFailure::External(ExternalFailure {
        stage: ExternalStage::Allocation,
        detail: reason.to_string(),
    })
}

/// A device allocation failure, preserving the memory domain's capacity fact.
fn allocation_failure(error: crate::Error) -> ExecutionFailure {
    match error {
        crate::Error::Capacity { required, available } => ExecutionFailure::External(
            ExternalFailure {
                stage: ExternalStage::Allocation,
                detail: format!(
                    "{required} bytes required, {available} charged bytes available"
                ),
            },
        ),
        error @ (crate::Error::LimitBelowCharges { .. }
        | crate::Error::Range { .. }
        | crate::Error::External(_)) => ExecutionFailure::External(ExternalFailure {
            stage: ExternalStage::Allocation,
            detail: error.to_string(),
        }),
    }
}

/// Encode one bound scalar in its ABI representation. A value with no word in
/// the representation is an invocation fact, reported with its reason.
fn encode_scalar(
    slot: ScalarSlot,
    dtype: DType,
    value: f64,
) -> Result<ScalarWord, InvalidInvocation> {
    let unrepresentable = |reason: String| {
        InvalidInvocation::ScalarRepresentation {
            slot,
            reason: format!("{value} is not representable as {reason}"),
        }
    };
    let integral = |value: f64| value.is_finite() && value.fract() == 0.0;
    Ok(match dtype {
        DType::Bool => {
            if !integral(value) || !(value == 0.0 || value == 1.0) {
                return Err(unrepresentable("a boolean word".into()));
            }
            ScalarWord {
                dtype,
                bits: value as u64,
            }
        }
        DType::I32 => {
            if !integral(value)
                || !(i32::MIN as f64 <= value && value <= i32::MAX as f64)
            {
                return Err(unrepresentable("a signed 32-bit integer".into()));
            }
            ScalarWord {
                dtype,
                bits: u64::from((value as i32) as u32),
            }
        }
        DType::U32 => {
            if !integral(value) || !(0.0 <= value && value <= u32::MAX as f64) {
                return Err(unrepresentable("an unsigned 32-bit integer".into()));
            }
            ScalarWord {
                dtype,
                bits: value as u64,
            }
        }
        DType::F32 => ScalarWord {
            dtype,
            bits: u64::from(f32::to_bits(value as f32)),
        },
        DType::F16 => ScalarWord {
            dtype,
            bits: u64::from(seismic_lang::numeric::f16_bits(value as f32)),
        },
        DType::BF16 => ScalarWord {
            dtype,
            bits: u64::from(
                f32::to_bits(seismic_lang::numeric::bf16_round(value as f32)) >> 16,
            ),
        },
    })
}
