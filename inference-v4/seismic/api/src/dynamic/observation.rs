//! Concrete invocation checks own private reference and native snapshots.
use super::*;
use seismic_compiler::numerics::{
    compare_outcome, Comparison, ComparisonError, ObservedInput, ObservedInvocation,
    ObservedResult, ObservedTensor, ObservedValue,
};
use seismic_lang::failure::SourceTermination;
use seismic_lang::interp::{Arg, Interpreter, OracleError, TensorData};

#[derive(Clone, Debug)]
pub struct CheckReport {
    pub status: &'static str,
    pub diagnostic: String,
    pub work_units: u64,
}
impl CheckReport {
    fn new(status: &'static str, message: impl ToString) -> Self {
        Self {
            status,
            diagnostic: message.to_string(),
            work_units: 0,
        }
    }
}
impl Kernel {
    pub fn check(
        &self,
        args: &[Value],
        policy: PrecisionPolicy,
        memory_bytes: u64,
        work_limit: u64,
    ) -> Result<CheckReport, Error> {
        self.function.validate_policy(&policy)?;
        if matches!(policy, PrecisionPolicy::Unconstrained) {
            return Err(Error::new(
                "ValueError",
                "a correctness check requires exact or bounded comparison",
            ));
        }
        if self.is_native() {
            return Ok(CheckReport::new(
                "unsupported",
                "direct-native preparation does not yet expose a bounded observation admission",
            ));
        }
        if args.len() != self.function.parameters().len() {
            return Err(Error::new("TypeError", "wrong argument count"));
        }
        let entry = self
            .function
            .module
            .checked
            .entry(
                self.function.info().id,
                &self.function.bindings(&self.elements)?,
            )
            .map_err(|e| Error::new("SourceError", e))?;
        let mut tensors = Vec::new();
        for v in args {
            collect_tensors(v, &mut tensors);
        }
        let mut storages: Vec<_> = tensors.iter().map(|t| t.0.storage.clone()).collect();
        storages.sort_by_key(|s| Arc::as_ptr(s) as usize);
        if storages.windows(2).any(|v| Arc::ptr_eq(&v[0], &v[1])) {
            return Ok(CheckReport::new(
                "unsupported",
                "aliased input snapshots are not representable by this observer",
            ));
        }
        let _guards: Vec<_> = storages.iter().map(|s| lock(&s.gate)).collect();
        let mut encoded = runtime::EncodedArgs::new();
        let mut intents = Vec::new();
        for ((_, ty), v) in self.function.parameters().iter().zip(args) {
            encode(ty, v, &self.elements, &mut encoded, &mut intents)?;
        }
        let mut base = 0u64;
        for t in tensors {
            let t = t.raw()?;
            if t.element().dtype().is_none() {
                return Ok(CheckReport::new(
                    "unsupported",
                    "encoded input snapshots require an authored decode",
                ));
            }
            // Native input, restore/readback image, reference construction, and final observation.
            base = base
                .saturating_add(t.byte_len().saturating_mul(4))
                .saturating_add((t.extents().len() as u64).saturating_mul(32));
            if t.extents().iter().fold(1u64, |a, b| a.saturating_mul(*b)) > work_limit {
                return Ok(CheckReport::new(
                    "resource_limit",
                    "input exceeds reference work budget",
                ));
            }
        }
        if base >= memory_bytes {
            return Ok(CheckReport::new(
                "resource_limit",
                "input snapshots exceed memory budget",
            ));
        }
        let mut interpreter = Interpreter::new(&entry);
        let mut oracle_args = Vec::new();
        let mut final_inputs = Vec::new();
        let mut ordinal = 0usize;
        let private = args
            .iter()
            .map(|v| {
                snapshot(
                    v,
                    &mut interpreter,
                    &mut oracle_args,
                    &mut final_inputs,
                    &mut ordinal,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reference = match interpreter.run_bounded_with_memory(
            &oracle_args,
            work_limit,
            memory_bytes - base,
        ) {
            Ok(r) => r,
            Err(e @ (OracleError::WorkLimit { .. } | OracleError::MemoryLimit { .. })) => {
                return Ok(CheckReport::new("resource_limit", e))
            }
            Err(OracleError::InterpreterDefect(e)) => return Err(Error::new("InternalError", e)),
            Err(OracleError::InvalidInvocation(e)) => {
                return Ok(CheckReport::new(
                    "unsupported",
                    format!("reference did not produce a comparable complete outcome: {e}"),
                ))
            }
        };
        let mut live = base.saturating_add(reference.retained_payload_bytes());
        // Reserve output readbacks before native allocation/admission.
        for output in reference.results() {
            if let seismic_lang::interp::OutcomeValue::Tensor(t) = output.value() {
                live = live
                    .saturating_add(
                        t.canonical_byte_len()
                            .map_err(|e| Error::new("InternalError", e))?
                            as u64,
                    )
                    .saturating_add((t.shape().len() * 8) as u64);
            }
        }
        if live >= memory_bytes {
            return Ok(CheckReport::new(
                "resource_limit",
                "reference and observations exceed memory budget",
            ));
        }
        let result = match self.call_outcome_limited(&private, Some(memory_bytes - live)) {
            Ok(v) => v,
            Err(e) if e.kind == "ResourceLimit" => {
                return Ok(CheckReport::new("resource_limit", e))
            }
            Err(e) => return Err(e),
        };
        let mut leaves = Vec::new();
        if let SourceTermination::Returned(value) = &result {
            flatten(value, &mut leaves);
        }
        let results = self
            .function
            .info()
            .results
            .iter()
            .zip(leaves)
            .map(|(info, v)| {
                Ok(ObservedResult {
                    path: info.path.clone(),
                    value: observe(v)?,
                })
            })
            .collect::<Result<_, Error>>()?;
        let inputs = final_inputs
            .iter()
            .map(|(ordinal, t)| {
                Ok(ObservedInput {
                    ordinal: *ordinal,
                    tensor: observe_tensor(t)?,
                })
            })
            .collect::<Result<_, Error>>()?;
        let mut remaining = work_limit.saturating_sub(reference.work_units());
        let comparison = compare_outcome(
            &reference,
            &ObservedInvocation {
                termination: match result {
                    SourceTermination::Returned(_) => SourceTermination::Returned(results),
                    SourceTermination::Failed(failure) => {
                        SourceTermination::Failed(failure.failure)
                    }
                },
                inputs,
            },
            &policy,
            &mut |cost| {
                remaining = remaining.checked_sub(cost).ok_or_else(|| {
                    ComparisonError::Resource("comparison work budget exceeded".into())
                })?;
                Ok(())
            },
        );
        let mut report = match comparison {
            Ok(Comparison::Match) => CheckReport::new(
                "passed",
                "complete invocation matches checked-source semantics",
            ),
            Ok(Comparison::Mismatch(d)) => CheckReport::new("failed", format!("{d:?}")),
            Ok(Comparison::Unsupported(e)) => CheckReport::new("unsupported", e),
            Err(ComparisonError::Resource(e)) => CheckReport::new("resource_limit", e),
            Err(ComparisonError::Contract(e)) => return Err(Error::new("InternalError", e)),
        };
        report.work_units = work_limit - remaining;
        Ok(report)
    }
}
fn snapshot(
    v: &Value,
    interpreter: &mut Interpreter<'_>,
    oracle: &mut Vec<Arg>,
    inputs: &mut Vec<(usize, super::super::Tensor)>,
    ordinal: &mut usize,
) -> Result<Value, Error> {
    Ok(match v {
        Value::Unit => Value::Unit,
        Value::Tuple(v) => Value::Tuple(
            v.iter()
                .map(|v| snapshot(v, interpreter, oracle, inputs, ordinal))
                .collect::<Result<_, _>>()?,
        ),
        Value::Tensor(t) | Value::Move(t) => {
            let raw = t.raw()?;
            let bytes = raw.read_to_host()?;
            let data = TensorData::dense_from_bytes(
                raw.element().dtype().expect("dense checked"),
                raw.extents().iter().map(|n| *n as usize).collect(),
                bytes.clone(),
            )
            .map_err(|e| Error::new("TensorError", e))?;
            oracle.push(Arg::Tensor(interpreter.add_tensor(data)));
            let private = Tensor::from_host(&raw.device(), raw.element(), raw.extents(), &bytes)?;
            inputs.push((*ordinal, private.raw()?));
            *ordinal += 1;
            if matches!(v, Value::Move(_)) {
                Value::Move(private)
            } else {
                Value::Tensor(private)
            }
        }
        Value::Scalar(s) => {
            use seismic_lang::reference_math::ReferenceScalar as R;
            oracle.push(match s.clone() {
                Scalar::Index(v) => Arg::Index(v),
                Scalar::Range(a, b) => Arg::Range(a, b),
                Scalar::F32(v) => Arg::Scalar(R::from_bits(DType::F32, v)),
                Scalar::F16(v) => Arg::Scalar(R::from_bits(DType::F16, v as u32)),
                Scalar::BF16(v) => Arg::Scalar(R::from_bits(DType::BF16, v as u32)),
                Scalar::I32(v) => Arg::Scalar(R::from_bits(DType::I32, v as u32)),
                Scalar::U32(v) => Arg::Scalar(R::from_bits(DType::U32, v)),
                Scalar::Bool(v) => Arg::Scalar(R::from_bits(DType::Bool, u32::from(v))),
            });
            *ordinal += 1;
            Value::Scalar(s.clone())
        }
    })
}
fn flatten<'a>(v: &'a Value, out: &mut Vec<&'a Value>) {
    match v {
        Value::Unit => {}
        Value::Tuple(v) => {
            for x in v {
                flatten(x, out)
            }
        }
        _ => out.push(v),
    }
}
fn observe_tensor(t: &super::super::Tensor) -> Result<ObservedTensor, Error> {
    Ok(ObservedTensor {
        representation: t.element().id(),
        shape: t.extents().iter().map(|v| *v as usize).collect(),
        bytes: t.read_to_host()?,
    })
}
fn observe(v: &Value) -> Result<ObservedValue, Error> {
    Ok(match v {
        Value::Tensor(t) => ObservedValue::Tensor(observe_tensor(&t.raw()?)?),
        Value::Scalar(Scalar::Range(a, b)) => ObservedValue::Range {
            start: seismic_lang::expr::SymbolValue::Nat(a.clone()),
            end: seismic_lang::expr::SymbolValue::Nat(b.clone()),
        },
        Value::Scalar(s @ Scalar::Index(_)) => ObservedValue::Index(s.symbol()),
        Value::Scalar(s) => ObservedValue::Scalar(s.symbol()),
        _ => return Err(Error::new("InternalError", "invalid observation leaf")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dynamic_check_compares_failed_source_state_without_mutating_callers() {
        let catalog = DeviceCatalog::discover().unwrap();
        let device = catalog.open_backend(BackendName::Cpu).unwrap();
        let module=Module::source(
            "fn probe(dst: &mut tensor[2] i32, divisor: i32):\n    dst[0] = 7\n    let unused = 42 / divisor\n    dst[1] = 9\n",
            "dynamic-failed-outcome.seismic",false,None,
        ).unwrap();
        let kernel = module
            .function("probe")
            .unwrap()
            .prepare(
                &device,
                BTreeMap::new(),
                PreparationOptions::feedback(
                    PrecisionPolicy::Exact,
                    seismic_compiler::feedback::FeedbackOptions {
                        search_time: std::time::Duration::ZERO,
                        ..Default::default()
                    },
                ),
            )
            .unwrap();
        let original = [2i32, 3]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let dst = Tensor::from_host(&device, Element::dense(DType::I32), &[2], &original).unwrap();
        let report = kernel
            .check(
                &[Value::Tensor(dst.clone()), Value::Scalar(Scalar::I32(0))],
                PrecisionPolicy::Exact,
                16 * 1024 * 1024,
                10000,
            )
            .unwrap();
        assert_eq!(report.status, "passed", "{}", report.diagnostic);
        assert_eq!(dst.read().unwrap(), original);
        let error = kernel
            .call(&[Value::Tensor(dst.clone()), Value::Scalar(Scalar::I32(0))])
            .err()
            .unwrap();
        assert_eq!(error.kind, "ExecutionError");
        assert!(error.message.contains("integer division by zero"));
        assert_eq!(
            dst.read().unwrap(),
            [7i32, 3]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }
}
