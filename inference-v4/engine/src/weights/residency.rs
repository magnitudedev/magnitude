//! Import stored tensor bytes into retained Seismic tensors. Container parsing
//! stays in the engine; numerical conversion and external packet repacking use
//! generated Seismic entries exclusively.

use super::{
    descriptor::{Stored, StoredTensor, Transform, WeightDescriptor},
    Error,
};
use crate::kernels;
use seismic::{DType, Device, Element, Kernel, PrecisionPolicy, Tensor};
use std::{collections::HashMap, rc::Rc};

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

#[derive(Clone)]
pub struct ResidentWeight {
    descriptor: WeightDescriptor,
    tensor: Tensor,
}

impl ResidentWeight {
    pub(crate) fn belongs_to(&self, device: &Device) -> bool {
        self.tensor.belongs_to(device)
    }
    pub fn descriptor(&self) -> &WeightDescriptor {
        &self.descriptor
    }
    pub fn element(&self) -> Element {
        self.tensor.element()
    }
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
}

fn external_element(encoding: super::gguf::Encoding) -> Option<Element> {
    use super::gguf::Encoding;
    let name = match encoding {
        Encoding::Q4K => "gguf_q4_k",
        Encoding::Q5K => "gguf_q5_k",
        Encoding::Q6K => "gguf_q6_k",
        Encoding::Q8_0 => "gguf_q8_0",
        Encoding::Iq4Xs => "gguf_iq4_xs",
        Encoding::F32 | Encoding::F16 => return None,
    };
    Some(Element::named(name).expect("registered GGUF representation is missing"))
}

fn resident_element(encoding: super::gguf::Encoding) -> Option<Element> {
    use super::gguf::Encoding;
    let name = match encoding {
        Encoding::Q4K => "q4k",
        Encoding::Q5K => "q5k",
        Encoding::Q6K => "q6k",
        Encoding::Q8_0 => "q8g32s",
        Encoding::Iq4Xs => "iq4g32",
        Encoding::F32 | Encoding::F16 => return None,
    };
    Some(Element::named(name).expect("registered resident representation is missing"))
}

pub struct Importer {
    device: Rc<Device>,
    precision: PrecisionPolicy,
    dense: HashMap<(DType, DType), Kernel<kernels::import_weight::Entry>>,
    negative_exp: HashMap<(DType, DType), Kernel<kernels::import_weight_negative_exp::Entry>>,
    repack: HashMap<(Element, Element), Kernel<kernels::repack_weight::Entry>>,
}

impl Importer {
    pub fn new(device: Rc<Device>, precision: PrecisionPolicy) -> Self {
        Self {
            device,
            precision,
            dense: HashMap::new(),
            negative_exp: HashMap::new(),
            repack: HashMap::new(),
        }
    }

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

        let tensor = match stored {
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
                        .is_none_or(|extent| !extent.is_multiple_of(encoding.block_elements()))
                {
                    return Err(invalid("block weight geometry or transform is unsupported"));
                }
                let source_element = external_element(*encoding)
                    .ok_or_else(|| invalid("dense encoding cannot use packet repacking"))?;
                let destination_element = resident_element(*encoding).ok_or_else(|| {
                    invalid("dense encoding has no resident packet representation")
                })?;
                let blocks = count / encoding.block_elements() as usize;
                let expected = u64::try_from(blocks)
                    .ok()
                    .and_then(|blocks| blocks.checked_mul(encoding.block_bytes()))
                    .ok_or_else(|| invalid("block storage byte count overflow"))?;
                if *nbytes != expected {
                    return Err(invalid("block storage byte count mismatch"));
                }
                let bytes = source.read(
                    *offset,
                    usize::try_from(*nbytes)
                        .map_err(|_| invalid("block bytes exceed host address range"))?,
                )?;
                let source_tensor = Tensor::from_host(
                    &self.device,
                    source_element,
                    &[u64::try_from(count)
                        .map_err(|_| invalid("weight element count exceeds shape domain"))?],
                    &bytes,
                )
                .map_err(Error::from)?;
                let key = (source_element, destination_element);
                if !self.repack.contains_key(&key) {
                    let kernel = kernels::repack_weight::for_device_with(
                        &self.device,
                        self.precision.clone(),
                        kernels::repack_weight::Elements {
                            T: source_element,
                            U: destination_element,
                        },
                    )
                    .map_err(Error::from)?;
                    self.repack.insert(key, kernel);
                }
                self.repack
                    .get(&key)
                    .expect("repack kernel was inserted")
                    .call(kernels::repack_weight::Args {
                        source: &source_tensor,
                    })
                    .map_err(Error::from)?
                    .value
                    .reshape(&descriptor.shape)
                    .map_err(Error::from)?
            }
            Stored::Dense(stored) => {
                validate_dense(stored)?;
                if element_count(&stored.shape)? != count || !stored.dtype.is_float() {
                    return Err(invalid(
                        "stored dense weight does not match its logical descriptor",
                    ));
                }
                let source_element = Element::dense(stored.dtype);
                let source = Tensor::from_host(
                    &self.device,
                    source_element,
                    &[u64::try_from(count)
                        .map_err(|_| invalid("weight element count exceeds shape domain"))?],
                    &stored.read()?,
                )
                .map_err(Error::from)?;
                if stored.dtype == target && descriptor.transform == Transform::Identity {
                    source.reshape(&descriptor.shape).map_err(Error::from)?
                } else {
                    let target_element = Element::dense(target);
                    let mut result = Tensor::zeros(
                        &self.device,
                        target_element,
                        &[u64::try_from(count)
                            .map_err(|_| invalid("weight element count exceeds shape domain"))?],
                    )
                    .map_err(Error::from)?;
                    let key = (stored.dtype, target);
                    match descriptor.transform {
                        Transform::Identity => {
                            if !self.dense.contains_key(&key) {
                                let kernel = kernels::import_weight::for_device_with(
                                    &self.device,
                                    self.precision.clone(),
                                    kernels::import_weight::Elements {
                                        T: source_element,
                                        U: target_element,
                                    },
                                )
                                .map_err(Error::from)?;
                                self.dense.insert(key, kernel);
                            }
                            self.dense
                                .get(&key)
                                .expect("dense import kernel was inserted")
                                .call(kernels::import_weight::Args {
                                    source: &source,
                                    result: &mut result,
                                })
                                .map_err(Error::from)?;
                        }
                        Transform::NegativeExp => {
                            if !self.negative_exp.contains_key(&key) {
                                let kernel = kernels::import_weight_negative_exp::for_device_with(
                                    &self.device,
                                    self.precision.clone(),
                                    kernels::import_weight_negative_exp::Elements {
                                        T: source_element,
                                        U: target_element,
                                    },
                                )
                                .map_err(Error::from)?;
                                self.negative_exp.insert(key, kernel);
                            }
                            self.negative_exp
                                .get(&key)
                                .expect("negative-exp import kernel was inserted")
                                .call(kernels::import_weight_negative_exp::Args {
                                    source: &source,
                                    result: &mut result,
                                })
                                .map_err(Error::from)?;
                        }
                    }
                    result.reshape(&descriptor.shape).map_err(Error::from)?
                }
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
                        .is_none_or(|extent| !extent.is_multiple_of(64))
                {
                    return Err(invalid("unsupported affine weight geometry or transform"));
                }
                let bytes = pack_q4g64(descriptor, codes, scales, biases)?;
                Tensor::from_host(
                    &self.device,
                    Element::named("q4g64").expect("registered q4g64 representation is missing"),
                    &descriptor.shape,
                    &bytes,
                )
                .map_err(Error::from)?
            }
        };

        Ok(ResidentWeight {
            descriptor: descriptor.clone(),
            tensor,
        })
    }
}

fn pack_q4g64(
    descriptor: &WeightDescriptor,
    codes: &StoredTensor,
    scales: &StoredTensor,
    biases: &StoredTensor,
) -> Result<Vec<u8>, Error> {
    for tensor in [codes, scales, biases] {
        validate_dense(tensor)?;
    }
    let mut code_shape = descriptor.shape.clone();
    *code_shape.last_mut().expect("weight shape is nonempty") /= 8;
    let mut coefficient_shape = descriptor.shape.clone();
    *coefficient_shape
        .last_mut()
        .expect("weight shape is nonempty") /= 64;
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
    let code_bytes = codes.read()?;
    let scale_bytes = scales.read()?;
    let bias_bytes = biases.read()?;
    let groups = element_count(&descriptor.shape)? / 64;
    let capacity = groups
        .checked_mul(36)
        .ok_or_else(|| invalid("q4g64 packet size overflow"))?;
    let mut packets = Vec::new();
    packets
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("q4g64 packet allocation failed"))?;
    for group in 0..groups {
        let words = group * 32;
        let coefficient = group * 2;
        packets.extend_from_slice(&code_bytes[words..words + 32]);
        packets.extend_from_slice(&scale_bytes[coefficient..coefficient + 2]);
        packets.extend_from_slice(&bias_bytes[coefficient..coefficient + 2]);
    }
    Ok(packets)
}

fn element_count(shape: &[u64]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(
                usize::try_from(*extent)
                    .map_err(|_| invalid("weight dimension exceeds host address range"))?,
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
