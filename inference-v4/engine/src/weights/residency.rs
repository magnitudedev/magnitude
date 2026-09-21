//! Import stored tensor bytes into retained native allocations. Conversion and
//! numerical transforms execute only through checked Seismic programs. Import
//! kernels are compiled eagerly at first use inside model loading — before the
//! importing weight becomes resident — and are reused for every later import
//! of the same geometry; loading is the preparation phase of the model.
use super::{
    descriptor::{Stored, StoredTensor, Transform, WeightDescriptor},
    Error,
};
use crate::{
    execution,
    preparation::{PreparationSession, Program, Settings},
};
use seismic_lang::{
    logical::specialization::ShapeBinding,
    program::{compile, SourceFile},
    types::{DType, Elem},
};
use seismic_runtime::{invocation::Bindings, plan::CompiledPlan, Buffer, Device};
use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
};
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
#[derive(Clone)]
pub struct ResidentWeight {
    descriptor: WeightDescriptor,
    element: Elem,
    planes: BTreeMap<String, Buffer>,
}
impl ResidentWeight {
    pub(crate) fn belongs_to(&self, device: &Device) -> bool {
        self.planes.values().all(|buffer| buffer.belongs_to(device))
    }
    pub fn descriptor(&self) -> &WeightDescriptor {
        &self.descriptor
    }
    pub fn element(&self) -> &Elem {
        &self.element
    }
    pub fn plane(&self, name: &str) -> Option<&Buffer> {
        self.planes.get(name)
    }
}
/// The import entry and resident representation of a block encoding.
pub fn block_import(encoding: super::gguf::Encoding) -> Option<(&'static str, &'static str)> {
    use super::gguf::Encoding;
    match encoding {
        Encoding::Q4K => Some(("import_q4k", "q4k")),
        Encoding::Q5K => Some(("import_q5k", "q5k")),
        Encoding::Q6K => Some(("import_q6k", "q6k")),
        Encoding::Q8_0 => Some(("import_q8_0", "q8g32s")),
        Encoding::Iq4Xs => Some(("import_iq4_xs", "iq4g32")),
        Encoding::F32 | Encoding::F16 => None,
    }
}
pub struct Importer {
    device: Rc<Device>,
    settings: Settings,
    program: Program,
    kernels: HashMap<(DType, DType, usize, bool), CompiledPlan>,
    block_kernels: HashMap<(super::gguf::Encoding, usize), CompiledPlan>,
}
impl Importer {
    pub fn new(device: Rc<Device>, settings: Settings) -> Result<Self, Error> {
        let program = compile(&[
            SourceFile {
                path: "gguf_import.seismic".into(),
                text: include_str!("../../../seismic-std/lib/kernels/gguf_import.seismic").into(),
            },
            SourceFile {
                path: "weight_import.seismic".into(),
                text: include_str!("../../../seismic-std/lib/kernels/weight_import.seismic").into(),
            },
        ])
        .map_err(|errors| {
            invalid(format!(
                "weight import program: {}",
                errors
                    .iter()
                    .map(|e| e.render())
                    .collect::<Vec<_>>()
                    .join("\n")
            ))
        })?;
        Ok(Self {
            device,
            settings,
            program,
            kernels: HashMap::new(),
            block_kernels: HashMap::new(),
        })
    }
    /// The target dtype applies to dense storage. Canonical affine planes retain
    /// their original packed representation and coefficient precision.
    pub fn import(
        &mut self,
        descriptor: &WeightDescriptor,
        stored: &Stored,
        target: DType,
    ) -> Result<ResidentWeight, Error> {
        let started = std::time::Instant::now();
        let elements = element_count(&descriptor.shape).unwrap_or(0) as u64;
        let result = self.import_inner(descriptor, stored, target);
        crate::telemetry::span_import(
            &descriptor.name,
            &match stored {
                Stored::GgmlBlocks { encoding, .. } => format!("{encoding:?}").to_lowercase(),
                Stored::Dense(_) => "dense".into(),
                Stored::AffinePlanes { .. } => "affine".into(),
            },
            elements,
            started.elapsed().as_secs_f64(),
        );
        result
    }

    fn import_inner(
        &mut self,
        descriptor: &WeightDescriptor,
        stored: &Stored,
        target: DType,
    ) -> Result<ResidentWeight, Error> {
        if !target.is_float() {
            return Err(invalid("weight target must be a floating dtype"));
        }
        let count = element_count(&descriptor.shape)?;
        if count == 0 {
            return Err(invalid("weight must contain at least one element"));
        }
        let (element, planes) = match stored {
            Stored::GgmlBlocks {
                source,
                offset,
                nbytes,
                shape,
                encoding,
            } => {
                if element_count(shape)? != count
                    || descriptor.transform != Transform::Identity
                    || shape
                        .last()
                        .is_none_or(|n| !n.is_multiple_of(encoding.block_elements()))
                {
                    return Err(invalid("block weight geometry or transform is unsupported"));
                }
                let (entry, representation) = block_import(*encoding)
                    .ok_or_else(|| invalid("dense encoding cannot use block import"))?;
                let repr = seismic_lang::repr::lookup(representation).unwrap();
                if descriptor
                    .shape
                    .last()
                    .is_none_or(|n| !n.is_multiple_of(u64::from(repr.storage_group())))
                {
                    return Err(invalid("resident packed row geometry is invalid"));
                }
                let blocks = count / encoding.block_elements() as usize;
                let expected = (blocks as u64)
                    .checked_mul(encoding.block_bytes())
                    .ok_or_else(|| invalid("block storage overflow"))?;
                if *nbytes != expected {
                    return Err(invalid("block storage byte count mismatch"));
                }
                let mut bytes = source.read(
                    *offset,
                    usize::try_from(*nbytes)
                        .map_err(|_| invalid("block bytes exceed host range"))?,
                )?;
                let padded = bytes
                    .len()
                    .checked_add(3)
                    .ok_or_else(|| invalid("block padding overflow"))?
                    / 4
                    * 4;
                bytes.resize(padded, 0);
                let input = self.device.buffer_from(&bytes).map_err(|e| invalid(e.to_string()))?;
                let key = (*encoding, blocks);
                if !self.block_kernels.contains_key(&key) {
                    let plan = self
                        .compile_kernel(
                            entry,
                            BTreeMap::from([("B".into(), ShapeBinding::Exact(blocks as u64))]),
                            BTreeMap::new(),
                        )
                        .map_err(|e| invalid(e.to_string()))?;
                    self.block_kernels.insert(key, plan);
                }
                let mut targets = BTreeMap::new();
                for plane in repr.planes() {
                    let size = plane
                        .bytes(count as u64)
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| invalid("packed plane byte size overflow"))?;
                    let buffer = self.device.buffer(size).map_err(|e| invalid(e.to_string()))?;
                    targets.insert(plane.name.to_string(), buffer);
                }
                let bindings = BlockImport {
                    input: &input,
                    planes: &targets,
                };
                let observed = execution::invoke_plan(
                    self.block_kernels.get(&key).expect("kernel prepared"),
                    &bindings,
                )
                .and_then(execution::run_observed)
                .map_err(|e| invalid(e.to_string()))?;
                for launch in &observed.1.launches {
                    crate::telemetry::span_launch(launch);
                }
                (Elem::Repr(representation.into()), targets)
            }
            Stored::Dense(tensor) => {
                validate_dense(tensor)?;
                if element_count(&tensor.shape)? != count || !tensor.dtype.is_float() {
                    return Err(invalid(
                        "stored dense weight does not match its logical descriptor",
                    ));
                }
                let input = self.device.buffer_from(&tensor.read()?).map_err(|e| invalid(e.to_string()))?;
                let output = if tensor.dtype == target
                    && descriptor.transform == Transform::Identity
                {
                    input
                } else {
                    let negative_exp = descriptor.transform == Transform::NegativeExp;
                    let key = (tensor.dtype, target, count, negative_exp);
                    if !self.kernels.contains_key(&key) {
                        let entry = if negative_exp {
                            "import_weight_negative_exp"
                        } else {
                            "import_weight"
                        };
                        let plan = self
                            .compile_kernel(
                                entry,
                                BTreeMap::from([("N".into(), ShapeBinding::Exact(count as u64))]),
                                BTreeMap::from([
                                    ("T".into(), Elem::Dtype(tensor.dtype)),
                                    ("U".into(), Elem::Dtype(target)),
                                ]),
                            )
                            .map_err(|e| invalid(e.to_string()))?;
                        self.kernels.insert(key, plan);
                    }
                    let bytes = count
                        .checked_mul(target.bytes() as usize)
                        .ok_or_else(|| invalid("resident weight size overflow"))?;
                    let output = self.device.buffer(bytes).map_err(|e| invalid(e.to_string()))?;
                    let bindings = DenseImport {
                        source: &input,
                        result: &output,
                    };
                    let observed = execution::invoke_plan(
                        self.kernels.get(&key).expect("kernel prepared"),
                        &bindings,
                    )
                    .and_then(execution::run_observed)
                    .map_err(|e| invalid(e.to_string()))?;
                    for launch in &observed.1.launches {
                        crate::telemetry::span_launch(launch);
                    }
                    output
                };
                (Elem::Dtype(target), BTreeMap::from([("".into(), output)]))
            }
            Stored::AffinePlanes {
                bits,
                group,
                codes,
                scales,
                biases,
            } => {
                if *bits != 4
                    || *group != 64
                    || descriptor.transform != Transform::Identity
                    || descriptor
                        .shape
                        .last()
                        .is_none_or(|n| !n.is_multiple_of(64))
                {
                    return Err(invalid("unsupported affine weight geometry or transform"));
                }
                for tensor in [codes, scales, biases] {
                    validate_dense(tensor)?;
                }
                let mut code_shape = descriptor.shape.clone();
                *code_shape.last_mut().unwrap() /= 8;
                let mut coefficient_shape = descriptor.shape.clone();
                *coefficient_shape.last_mut().unwrap() /= 64;
                if codes.dtype != DType::U32
                    || scales.dtype != DType::BF16
                    || biases.dtype != DType::BF16
                    || codes.shape != code_shape
                    || scales.shape != coefficient_shape
                    || biases.shape != coefficient_shape
                {
                    return Err(invalid(
                        "affine planes do not match canonical q4g64 geometry",
                    ));
                }
                let mut planes = BTreeMap::new();
                for (name, tensor) in [("words", codes), ("scale", scales), ("bias", biases)] {
                    planes.insert(
                        name.into(),
                        self.device.buffer_from(&tensor.read()?).map_err(|e| invalid(e.to_string()))?,
                    );
                }
                (Elem::Repr("q4g64".into()), planes)
            }
        };
        Ok(ResidentWeight {
            descriptor: descriptor.clone(),
            element,
            planes,
        })
    }
    fn compile_kernel(
        &self,
        entry: &str,
        shapes: BTreeMap<String, ShapeBinding>,
        elements: BTreeMap<String, Elem>,
    ) -> Result<CompiledPlan, crate::Error> {
        PreparationSession::new(&self.device, &self.program, self.settings.clone()).compile_entry(
            entry, shapes, elements,
        )
    }
}
struct BlockImport<'a> {
    input: &'a Buffer,
    planes: &'a BTreeMap<String, Buffer>,
}
impl Bindings for BlockImport<'_> {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        if !plane.is_empty() {
            return None;
        }
        match root {
            "data" | "halves" => Some(self.input),
            other => self.planes.get(other),
        }
    }
    fn scalar(&self, _name: &str) -> Option<f64> {
        None
    }
    fn shape(&self, _name: &str) -> Option<u64> {
        None
    }
}
struct DenseImport<'a> {
    source: &'a Buffer,
    result: &'a Buffer,
}
impl Bindings for DenseImport<'_> {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        if !plane.is_empty() {
            return None;
        }
        match root {
            "source" => Some(self.source),
            "result" => Some(self.result),
            _ => None,
        }
    }
    fn scalar(&self, _name: &str) -> Option<f64> {
        None
    }
    fn shape(&self, _name: &str) -> Option<u64> {
        None
    }
}
fn element_count(shape: &[u64]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |n, d| {
        n.checked_mul(
            usize::try_from(*d).map_err(|_| invalid("weight dimension exceeds address range"))?,
        )
        .ok_or_else(|| invalid("weight element count overflow"))
    })
}
fn validate_dense(tensor: &StoredTensor) -> Result<(), Error> {
    let bytes = element_count(&tensor.shape)?
        .checked_mul(tensor.dtype.bytes() as usize)
        .ok_or_else(|| invalid("stored weight byte count overflow"))?;
    if u64::try_from(bytes).ok() != Some(tensor.nbytes) {
        return Err(invalid(
            "stored tensor byte count does not match shape and dtype",
        ));
    }
    Ok(())
}
