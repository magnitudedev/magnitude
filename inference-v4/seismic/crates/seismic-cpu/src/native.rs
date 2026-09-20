//! Native encoding of the resolved executable contract.

use crate::{
    NativeExecution, NativePhase,
    physical::{CpuDialect, EncodedLaunch},
};
use cranelift_codegen::ir::{
    self, AbiParam, InstBuilder, MemFlags, Value,
    condcodes::{FloatCC, IntCC},
    types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use seismic_compiler::pipeline::{EncodedPlan, EncodedScheduleItem};
use seismic_compiler::terminal::{
    ResolvedScalarInstructionKind as I, ScalarIndex, ScalarLiteral, ScalarValueId,
};
use seismic_lang::{
    logical::Type,
    syntax::ast::{BinaryOp, UnaryOp},
    types::DType,
};
use seismic_realization::{
    BufferRole, BufferSpec, InvocationConditions,
    executable::{AbiRole, ResolvedLaunch, ResolvedScheduleItem, ResolvedStorageId, StorageScope},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
struct TensorValue {
    planes: Vec<TensorPlane>,
}

#[derive(Clone)]
struct TensorPlane {
    pointer: Value,
    shape: Vec<u64>,
    byte_strides: Vec<u64>,
    dtype: DType,
    representation: Option<String>,
    plane: Vec<String>,
}

fn cpu_type(dtype: DType) -> ir::Type {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => types::F32,
        DType::I32 | DType::U32 => types::I64,
        DType::Bool => types::I8,
    }
}

fn load_scalar(builder: &mut FunctionBuilder<'_>, address: Value, dtype: DType) -> Value {
    match dtype {
        DType::BF16 => {
            let bits = builder
                .ins()
                .load(types::I16, MemFlags::trusted(), address, 0);
            let bits = builder.ins().uextend(types::I32, bits);
            let bits = builder.ins().ishl_imm(bits, 16);
            builder.ins().bitcast(types::F32, MemFlags::new(), bits)
        }
        DType::F16 => {
            let value = builder
                .ins()
                .load(types::F16, MemFlags::trusted(), address, 0);
            builder.ins().fpromote(types::F32, value)
        }
        _ => builder
            .ins()
            .load(cpu_type(dtype), MemFlags::trusted(), address, 0),
    }
}

fn store_scalar(builder: &mut FunctionBuilder<'_>, address: Value, dtype: DType, value: Value) {
    let value = match dtype {
        DType::BF16 => {
            let bits = builder.ins().bitcast(types::I32, MemFlags::new(), value);
            let rounding = builder.ins().iadd_imm(bits, 0x7fff);
            let shifted = builder.ins().ushr_imm(bits, 16);
            let lsb = builder.ins().band_imm(shifted, 1);
            let rounded = builder.ins().iadd(rounding, lsb);
            let high = builder.ins().ushr_imm(rounded, 16);
            builder.ins().ireduce(types::I16, high)
        }
        DType::F16 => builder.ins().fdemote(types::F16, value),
        _ => value,
    };
    builder.ins().store(MemFlags::trusted(), value, address, 0);
}

fn first_storage(
    transport: &seismic_realization::executable::ResolvedValueTransport,
) -> Result<ResolvedStorageId, String> {
    match transport {
        seismic_realization::executable::ResolvedValueTransport::Storage(values) => {
            Ok(*values.iter().next().unwrap())
        }
        _ => Err("CPU scalar value is not storage-backed".into()),
    }
}

fn pointer(
    builder: &mut FunctionBuilder<'_>,
    table: Value,
    slots: &BTreeMap<ResolvedStorageId, usize>,
    storage: ResolvedStorageId,
) -> Result<Value, String> {
    let slot = slots
        .get(&storage)
        .ok_or_else(|| format!("CPU launch omits binding for storage#{}", storage.0))?;
    let address = builder.ins().iadd_imm(table, (*slot * 8) as i64);
    Ok(builder
        .ins()
        .load(types::I64, MemFlags::trusted(), address, 0))
}

fn tensor(
    builder: &mut FunctionBuilder<'_>,
    launch: &ResolvedLaunch<CpuDialect>,
    table: Value,
    slots: &BTreeMap<ResolvedStorageId, usize>,
    transport: &seismic_realization::executable::ResolvedValueTransport,
) -> Result<TensorValue, String> {
    let seismic_realization::executable::ResolvedValueTransport::Storage(ids) = transport else {
        return Err("CPU tensor is not storage-backed".into());
    };
    let mut planes = Vec::new();
    for id in ids.iter() {
        let storage = launch
            .storage
            .values()
            .find(|storage| storage.id == *id)
            .ok_or_else(|| format!("CPU storage#{} has no launch descriptor", id.0))?;
        let crate::physical::CpuResolvedLayout::TensorPlane {
            dtype,
            shape,
            representation,
            plane,
        } = &storage.layout
        else {
            return Err("CPU tensor transport names a scalar slot".into());
        };
        let byte_strides = match &storage.provenance.address {
            seismic_realization::executable::ResolvedPhysicalAddress::DenseAffine {
                byte_strides,
                ..
            } => byte_strides.clone(),
            seismic_realization::executable::ResolvedPhysicalAddress::Representation {
                logical_strides,
                ..
            } => logical_strides.clone(),
        };
        planes.push(TensorPlane {
            pointer: pointer(builder, table, slots, *id)?,
            shape: shape.clone(),
            byte_strides,
            dtype: *dtype,
            representation: representation.clone(),
            plane: plane.clone(),
        });
    }
    Ok(TensorValue { planes })
}

pub fn encode_launch(launch: &ResolvedLaunch<CpuDialect>) -> Result<NativePhase, String> {
    let call_conv = crate::codegen::Policy::host()?.call_conv();
    let mut signature = ir::Signature::new(call_conv);
    signature
        .params
        .extend((0..4).map(|_| AbiParam::new(types::I64)));
    signature.returns.push(AbiParam::new(types::I32));
    let mut function = ir::Function::with_name_signature(ir::UserFuncName::user(0, 0), signature);
    let mut context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut function, &mut context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    let table = builder.block_params(entry)[0];
    let item = builder.block_params(entry)[3];
    let mut groups = launch.binding_groups.clone();
    groups.sort_by_key(|group| group.slot);
    let bindings = groups
        .iter()
        .flat_map(|group| group.members.iter().map(|member| member.storage))
        .collect::<Vec<_>>();
    let slots = bindings
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, storage)| (storage, slot))
        .collect::<BTreeMap<_, _>>();
    let mut values: BTreeMap<ScalarValueId, (Value, DType)> = BTreeMap::new();
    let mut tuples: BTreeMap<ScalarValueId, Vec<ScalarValueId>> = BTreeMap::new();
    let mut tensors: BTreeMap<ScalarValueId, TensorValue> = BTreeMap::new();
    let mut places: BTreeMap<ScalarValueId, (Value, DType)> = BTreeMap::new();
    let mut imports = Vec::new();
    for step in launch.kernel.steps.iter() {
        if let seismic_realization::executable::ResolvedKernelStep::MappedTask {
            mapping,
            instructions,
            ..
        } = step
        {
            let mut axis_coordinates = BTreeMap::new();
            let mut serial_loops = Vec::new();
            for (axis, map) in mapping.axes.iter().enumerate() {
                let coordinate = match map {
                    seismic_realization::executable::ResolvedAxisMap::Serial { .. } => {
                        let header = builder.create_block();
                        let body = builder.create_block();
                        let exit = builder.create_block();
                        builder.append_block_param(header, types::I64);
                        let zero = builder.ins().iconst(types::I64, 0);
                        builder.ins().jump(header, &[zero.into()]);
                        builder.switch_to_block(header);
                        let coordinate = builder.block_params(header)[0];
                        let active = builder.ins().icmp_imm(
                            IntCC::UnsignedLessThan,
                            coordinate,
                            mapping.logical_extents[axis] as i64,
                        );
                        builder.ins().brif(active, body, &[], exit, &[]);
                        builder.switch_to_block(body);
                        builder.seal_block(body);
                        serial_loops.push((header, exit, coordinate));
                        coordinate
                    }
                    seismic_realization::executable::ResolvedAxisMap::Grid {
                        workgroup_axis,
                        participant_axis,
                        ..
                    } => {
                        let mut divisor = 1u64;
                        for value in launch.geometry.participants_per_workgroup
                            [..*participant_axis as usize]
                            .iter()
                            .chain(launch.geometry.workgroups[..*workgroup_axis as usize].iter())
                        {
                            divisor = divisor
                                .checked_mul(*value)
                                .ok_or("CPU grid coordinate divisor overflow")?;
                        }
                        let divided = if divisor == 1 {
                            item
                        } else {
                            builder.ins().udiv_imm(item, divisor as i64)
                        };
                        let extent = mapping.logical_extents[axis];
                        if extent == 0 {
                            builder.ins().iconst(types::I64, 0)
                        } else {
                            builder.ins().urem_imm(divided, extent as i64)
                        }
                    }
                    seismic_realization::executable::ResolvedAxisMap::SubgroupLane { .. } => {
                        builder.ins().band_imm(item, 31)
                    }
                };
                axis_coordinates.insert(axis as u32, coordinate);
            }
            for instruction in instructions.iter() {
                match instruction {
                    crate::physical::CpuResolvedInstruction::Coordinate {
                        value,
                        axis,
                        component,
                        components,
                    } => {
                        let mut coordinate = *axis_coordinates
                            .get(axis)
                            .ok_or("CPU coordinate axis absent")?;
                        let trailing = components[*component as usize + 1..]
                            .iter()
                            .try_fold(1u64, |product, extent| product.checked_mul(*extent))
                            .ok_or("CPU coordinate component stride overflow")?;
                        if trailing != 1 {
                            coordinate = builder.ins().udiv_imm(coordinate, trailing as i64);
                        }
                        let extent = components[*component as usize];
                        if extent != 0 {
                            coordinate = builder.ins().urem_imm(coordinate, extent as i64);
                        }
                        values.insert(*value, (coordinate, DType::I32));
                    }
                    crate::physical::CpuResolvedInstruction::Operand {
                        value,
                        ty,
                        transport,
                        ..
                    } => {
                        if matches!(ty, Type::Scalar(_) | Type::Index { .. }) {
                            let dtype = match ty {
                                Type::Scalar(dtype) => *dtype,
                                _ => DType::I32,
                            };
                            let storage = first_storage(transport)?;
                            let address = pointer(&mut builder, table, &slots, storage)?;
                            let native = load_scalar(&mut builder, address, dtype);
                            values.insert(*value, (native, dtype));
                        } else if matches!(ty, Type::Tensor(_)) {
                            tensors.insert(
                                *value,
                                tensor(&mut builder, launch, table, &slots, transport)?,
                            );
                        }
                    }
                    crate::physical::CpuResolvedInstruction::Output {
                        value,
                        ty,
                        transport,
                        ..
                    } => {
                        if matches!(ty, Type::Scalar(_) | Type::Index { .. }) {
                            let (native, dtype) = *values
                                .get(value)
                                .ok_or_else(|| format!("CPU output value#{} is absent", value.0))?;
                            let storage = first_storage(transport)?;
                            let address = pointer(&mut builder, table, &slots, storage)?;
                            store_scalar(&mut builder, address, dtype, native);
                        }
                    }
                    crate::physical::CpuResolvedInstruction::Scalar { scalar, .. } => {
                        let result = scalar.result;
                        let scalar_value =
                            |values: &BTreeMap<ScalarValueId, (Value, DType)>,
                             id: ScalarValueId| {
                                values
                                    .get(&id)
                                    .copied()
                                    .ok_or_else(|| format!("CPU scalar value#{} is absent", id.0))
                            };
                        match &scalar.kind {
                            I::NoOp => {}
                            I::Literal(literal) => {
                                let id = result.ok_or("CPU literal has no result")?;
                                let value = match literal {
                                    ScalarLiteral::Int(value) => {
                                        (builder.ins().iconst(types::I64, *value), DType::I32)
                                    }
                                    ScalarLiteral::Float(bits) => (
                                        builder.ins().f32const(ir::immediates::Ieee32::with_bits(
                                            (f64::from_bits(*bits) as f32).to_bits(),
                                        )),
                                        DType::F32,
                                    ),
                                    ScalarLiteral::Bool(value) => (
                                        builder.ins().iconst(types::I8, i64::from(*value)),
                                        DType::Bool,
                                    ),
                                };
                                values.insert(id, value);
                            }
                            I::Shape(value) => {
                                values.insert(
                                    result.ok_or("CPU shape has no result")?,
                                    (builder.ins().iconst(types::I64, *value), DType::I32),
                                );
                            }
                            I::Tuple(fields) => {
                                tuples.insert(
                                    result.ok_or("CPU tuple has no result")?,
                                    fields.clone(),
                                );
                            }
                            I::Range(lo, hi) => {
                                tuples.insert(
                                    result.ok_or("CPU range has no result")?,
                                    vec![*lo, *hi],
                                );
                            }
                            I::Field { value, index } => {
                                let field = *tuples
                                    .get(value)
                                    .and_then(|fields| fields.get(*index))
                                    .ok_or("CPU aggregate field is absent")?;
                                values.insert(
                                    result.ok_or("CPU field has no result")?,
                                    scalar_value(&values, field)?,
                                );
                            }
                            I::Storage {
                                operation,
                                bindings,
                                inputs,
                                fill_bits,
                            } => {
                                let mut ids = bindings.iter().map(|binding| binding.storage);
                                let mut bundle = seismic_realization::executable::NonEmpty::new(
                                    ids.next().ok_or("CPU storage instruction has no plane")?,
                                );
                                for id in ids {
                                    bundle.push(id);
                                }
                                let transport = seismic_realization::executable::ResolvedValueTransport::Storage(bundle);
                                let tensor =
                                    tensor(&mut builder, launch, table, &slots, &transport)?;
                                if matches!(
                                    operation,
                                    seismic_compiler::terminal::StorageOperation::Fill
                                ) {
                                    let bits =
                                        fill_bits.ok_or("CPU fill storage has no fill value")?;
                                    for plane in &tensor.planes {
                                        let elements = plane
                                            .shape
                                            .iter()
                                            .try_fold(1u64, |n, extent| n.checked_mul(*extent))
                                            .ok_or("CPU fill extent overflow")?;
                                        let header = builder.create_block();
                                        let body = builder.create_block();
                                        let exit = builder.create_block();
                                        builder.append_block_param(header, types::I64);
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        builder.ins().jump(header, &[zero.into()]);
                                        builder.switch_to_block(header);
                                        let index = builder.block_params(header)[0];
                                        let active = builder.ins().icmp_imm(
                                            IntCC::UnsignedLessThan,
                                            index,
                                            elements as i64,
                                        );
                                        builder.ins().brif(active, body, &[], exit, &[]);
                                        builder.switch_to_block(body);
                                        builder.seal_block(body);
                                        let address = builder
                                            .ins()
                                            .imul_imm(index, i64::from(plane.dtype.bytes()));
                                        let address = builder.ins().iadd(plane.pointer, address);
                                        let value = match plane.dtype {
                                            DType::F16 | DType::BF16 => {
                                                builder.ins().iconst(types::I16, bits as i64)
                                            }
                                            DType::F32 => builder.ins().f32const(
                                                ir::immediates::Ieee32::with_bits(bits as u32),
                                            ),
                                            _ => builder
                                                .ins()
                                                .iconst(cpu_type(plane.dtype), bits as i64),
                                        };
                                        builder.ins().store(MemFlags::trusted(), value, address, 0);
                                        let next = builder.ins().iadd_imm(index, 1);
                                        builder.ins().jump(header, &[next.into()]);
                                        builder.seal_block(header);
                                        builder.switch_to_block(exit);
                                        builder.seal_block(exit);
                                    }
                                } else if matches!(
                                    operation,
                                    seismic_compiler::terminal::StorageOperation::Snapshot
                                        | seismic_compiler::terminal::StorageOperation::Materialize
                                ) && !inputs.is_empty()
                                {
                                    let source = tensors
                                        .get(&inputs[0])
                                        .ok_or("CPU storage copy source is not a tensor")?;
                                    if source.planes.len() != tensor.planes.len() {
                                        return Err("CPU storage copy plane counts differ".into());
                                    }
                                    for (source, destination) in
                                        source.planes.iter().zip(&tensor.planes)
                                    {
                                        if source.dtype != destination.dtype
                                            || source.shape != destination.shape
                                        {
                                            return Err("CPU storage copy layouts differ".into());
                                        }
                                        let elements = source
                                            .shape
                                            .iter()
                                            .try_fold(1u64, |n, extent| n.checked_mul(*extent))
                                            .ok_or("CPU copy extent overflow")?;
                                        let header = builder.create_block();
                                        let body = builder.create_block();
                                        let exit = builder.create_block();
                                        builder.append_block_param(header, types::I64);
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        builder.ins().jump(header, &[zero.into()]);
                                        builder.switch_to_block(header);
                                        let index = builder.block_params(header)[0];
                                        let active = builder.ins().icmp_imm(
                                            IntCC::UnsignedLessThan,
                                            index,
                                            elements as i64,
                                        );
                                        builder.ins().brif(active, body, &[], exit, &[]);
                                        builder.switch_to_block(body);
                                        builder.seal_block(body);
                                        let offset = builder
                                            .ins()
                                            .imul_imm(index, i64::from(source.dtype.bytes()));
                                        let source_address =
                                            builder.ins().iadd(source.pointer, offset);
                                        let destination_address =
                                            builder.ins().iadd(destination.pointer, offset);
                                        let value =
                                            load_scalar(&mut builder, source_address, source.dtype);
                                        store_scalar(
                                            &mut builder,
                                            destination_address,
                                            destination.dtype,
                                            value,
                                        );
                                        let next = builder.ins().iadd_imm(index, 1);
                                        builder.ins().jump(header, &[next.into()]);
                                        builder.seal_block(header);
                                        builder.switch_to_block(exit);
                                        builder.seal_block(exit);
                                    }
                                }
                                tensors.insert(
                                    result.ok_or("CPU storage instruction has no result")?,
                                    tensor,
                                );
                            }
                            I::View { base, .. } => {
                                let value = tensors
                                    .get(base)
                                    .cloned()
                                    .ok_or("CPU view base is not a tensor")?;
                                tensors.insert(result.ok_or("CPU view has no result")?, value);
                            }
                            I::Index { base, indices } => {
                                let source = tensors
                                    .get(base)
                                    .cloned()
                                    .ok_or("CPU index base is not a tensor")?;
                                let mut output = Vec::new();
                                for mut plane in source.planes {
                                    let mut offset = builder.ins().iconst(types::I64, 0);
                                    let mut kept_shape = Vec::new();
                                    let mut kept_strides = Vec::new();
                                    for (axis, index) in indices.iter().enumerate() {
                                        match index {
                                            ScalarIndex::Point(value)
                                            | ScalarIndex::Coordinate(value) => {
                                                let (value, _) = scalar_value(&values, *value)?;
                                                let term = builder.ins().imul_imm(
                                                    value,
                                                    plane.byte_strides[axis] as i64,
                                                );
                                                offset = builder.ins().iadd(offset, term);
                                            }
                                            ScalarIndex::Slice(_) => {
                                                kept_shape.push(plane.shape[axis]);
                                                kept_strides.push(plane.byte_strides[axis]);
                                            }
                                            ScalarIndex::Range { start, .. } => {
                                                if let Some(start) = start {
                                                    let (start, _) = scalar_value(&values, *start)?;
                                                    let term = builder.ins().imul_imm(
                                                        start,
                                                        plane.byte_strides[axis] as i64,
                                                    );
                                                    offset = builder.ins().iadd(offset, term);
                                                }
                                                kept_shape.push(plane.shape[axis]);
                                                kept_strides.push(plane.byte_strides[axis]);
                                            }
                                        }
                                    }
                                    plane.pointer = builder.ins().iadd(plane.pointer, offset);
                                    plane.shape = kept_shape;
                                    plane.byte_strides = kept_strides;
                                    output.push(plane);
                                }
                                let id = result.ok_or("CPU index has no result")?;
                                if output.first().is_some_and(|plane| plane.shape.is_empty()) {
                                    let plane = output.first().unwrap();
                                    places.insert(id, (plane.pointer, plane.dtype));
                                    let value =
                                        load_scalar(&mut builder, plane.pointer, plane.dtype);
                                    values.insert(id, (value, plane.dtype));
                                } else {
                                    tensors.insert(id, TensorValue { planes: output });
                                }
                            }
                            I::Cast { dtype, value } => {
                                let (value, from) = scalar_value(&values, *value)?;
                                let native = match (
                                    cpu_type(from).is_float(),
                                    cpu_type(*dtype).is_float(),
                                ) {
                                    (true, true) | (false, false) => value,
                                    (true, false) => {
                                        builder.ins().fcvt_to_sint_sat(types::I64, value)
                                    }
                                    (false, true) => {
                                        builder.ins().fcvt_from_sint(types::F32, value)
                                    }
                                };
                                values.insert(
                                    result.ok_or("CPU cast has no result")?,
                                    (native, *dtype),
                                );
                            }
                            I::Unary { op, value } => {
                                let (value, dtype) = scalar_value(&values, *value)?;
                                let native = match op {
                                    UnaryOp::Neg if cpu_type(dtype).is_float() => {
                                        builder.ins().fneg(value)
                                    }
                                    UnaryOp::Neg => builder.ins().ineg(value),
                                    UnaryOp::Not => builder.ins().icmp_imm(IntCC::Equal, value, 0),
                                    UnaryOp::BitNot => builder.ins().bnot(value),
                                };
                                values.insert(
                                    result.ok_or("CPU unary has no result")?,
                                    (
                                        native,
                                        if matches!(op, UnaryOp::Not) {
                                            DType::Bool
                                        } else {
                                            dtype
                                        },
                                    ),
                                );
                            }
                            I::Binary { op, lhs, rhs } => {
                                let (lhs, dtype) = scalar_value(&values, *lhs)?;
                                let (rhs, other) = scalar_value(&values, *rhs)?;
                                if dtype != other {
                                    return Err("CPU binary operand types differ".into());
                                }
                                let float = cpu_type(dtype).is_float();
                                let native = match (op, float) {
                                    (BinaryOp::Add, true) => builder.ins().fadd(lhs, rhs),
                                    (BinaryOp::Sub, true) => builder.ins().fsub(lhs, rhs),
                                    (BinaryOp::Mul, true) => builder.ins().fmul(lhs, rhs),
                                    (BinaryOp::Div, true) => builder.ins().fdiv(lhs, rhs),
                                    (BinaryOp::Add, false) => builder.ins().iadd(lhs, rhs),
                                    (BinaryOp::Sub, false) => builder.ins().isub(lhs, rhs),
                                    (BinaryOp::Mul, false) => builder.ins().imul(lhs, rhs),
                                    (BinaryOp::Div, false) => builder.ins().sdiv(lhs, rhs),
                                    (BinaryOp::Rem, false) => builder.ins().srem(lhs, rhs),
                                    (BinaryOp::BitOr | BinaryOp::Or, false) => {
                                        builder.ins().bor(lhs, rhs)
                                    }
                                    (BinaryOp::BitXor, false) => builder.ins().bxor(lhs, rhs),
                                    (BinaryOp::BitAnd | BinaryOp::And, false) => {
                                        builder.ins().band(lhs, rhs)
                                    }
                                    (BinaryOp::Shl, false) => builder.ins().ishl(lhs, rhs),
                                    (BinaryOp::Shr, false) => builder.ins().sshr(lhs, rhs),
                                    (comparison, true) => builder.ins().fcmp(
                                        match comparison {
                                            BinaryOp::Eq => FloatCC::Equal,
                                            BinaryOp::Ne => FloatCC::NotEqual,
                                            BinaryOp::Lt => FloatCC::LessThan,
                                            BinaryOp::Le => FloatCC::LessThanOrEqual,
                                            BinaryOp::Gt => FloatCC::GreaterThan,
                                            BinaryOp::Ge => FloatCC::GreaterThanOrEqual,
                                            _ => {
                                                return Err(format!(
                                                    "CPU float binary {comparison:?} is invalid"
                                                ));
                                            }
                                        },
                                        lhs,
                                        rhs,
                                    ),
                                    (comparison, false) => builder.ins().icmp(
                                        match comparison {
                                            BinaryOp::Eq => IntCC::Equal,
                                            BinaryOp::Ne => IntCC::NotEqual,
                                            BinaryOp::Lt => IntCC::SignedLessThan,
                                            BinaryOp::Le => IntCC::SignedLessThanOrEqual,
                                            BinaryOp::Gt => IntCC::SignedGreaterThan,
                                            BinaryOp::Ge => IntCC::SignedGreaterThanOrEqual,
                                            _ => {
                                                return Err(format!(
                                                    "CPU integer binary {comparison:?} is invalid"
                                                ));
                                            }
                                        },
                                        lhs,
                                        rhs,
                                    ),
                                };
                                let out = if matches!(
                                    op,
                                    BinaryOp::Eq
                                        | BinaryOp::Ne
                                        | BinaryOp::Lt
                                        | BinaryOp::Le
                                        | BinaryOp::Gt
                                        | BinaryOp::Ge
                                ) {
                                    DType::Bool
                                } else {
                                    dtype
                                };
                                values.insert(
                                    result.ok_or("CPU binary has no result")?,
                                    (native, out),
                                );
                            }
                            I::Select {
                                condition,
                                then_value,
                                else_value,
                            } => {
                                let (condition, _) = scalar_value(&values, *condition)?;
                                let (then_value, dtype) = scalar_value(&values, *then_value)?;
                                let (else_value, other) = scalar_value(&values, *else_value)?;
                                if dtype != other {
                                    return Err("CPU select branch types differ".into());
                                }
                                values.insert(
                                    result.ok_or("CPU select has no result")?,
                                    (
                                        builder.ins().select(condition, then_value, else_value),
                                        dtype,
                                    ),
                                );
                            }
                            I::Extent { base, axis } => {
                                let extent = *tensors
                                    .get(base)
                                    .and_then(|tensor| tensor.planes.first())
                                    .and_then(|plane| plane.shape.get(*axis))
                                    .ok_or("CPU extent axis is absent")?;
                                values.insert(
                                    result.ok_or("CPU extent has no result")?,
                                    (builder.ins().iconst(types::I64, extent as i64), DType::I32),
                                );
                            }
                            I::Accessor { base, name } => {
                                let source = tensors
                                    .get(base)
                                    .ok_or("CPU accessor base is not a packed tensor")?;
                                let plane = source
                                    .planes
                                    .iter()
                                    .find(|plane| {
                                        plane.plane.last().is_some_and(|part| part == name)
                                    })
                                    .cloned()
                                    .ok_or_else(|| {
                                        format!("CPU packed representation has no `{name}` plane")
                                    })?;
                                tensors.insert(
                                    result.ok_or("CPU accessor has no result")?,
                                    TensorValue {
                                        planes: vec![plane],
                                    },
                                );
                            }
                            I::Geometry { base, axis, valid } => {
                                let extent = *tensors
                                    .get(base)
                                    .and_then(|tensor| tensor.planes.first())
                                    .and_then(|plane| plane.shape.get(*axis))
                                    .ok_or("CPU geometry axis is absent")?;
                                let pair = if *valid {
                                    (
                                        builder.ins().iconst(types::I8, i64::from(extent != 0)),
                                        DType::Bool,
                                    )
                                } else {
                                    (builder.ins().iconst(types::I64, extent as i64), DType::I32)
                                };
                                values.insert(result.ok_or("CPU geometry has no result")?, pair);
                            }
                            I::Atomic { op, place, value } => {
                                // The correctness alternative maps every logical axis serially,
                                // so the selected launch has no concurrent participant touching
                                // this place. Its native atomic consequence is therefore exactly
                                // the corresponding load/operation/store sequence.
                                let (address, dtype) = *places
                                    .get(place)
                                    .ok_or("CPU atomic target is not an indexed place")?;
                                let (rhs, other) = scalar_value(&values, *value)?;
                                if dtype != other {
                                    return Err("CPU atomic types differ".into());
                                }
                                let lhs = load_scalar(&mut builder, address, dtype);
                                let native = match (op, cpu_type(dtype).is_float()) {
                                    (BinaryOp::Add, true) => builder.ins().fadd(lhs, rhs),
                                    (BinaryOp::Sub, true) => builder.ins().fsub(lhs, rhs),
                                    (BinaryOp::Mul, true) => builder.ins().fmul(lhs, rhs),
                                    (BinaryOp::Add, false) => builder.ins().iadd(lhs, rhs),
                                    (BinaryOp::Sub, false) => builder.ins().isub(lhs, rhs),
                                    (BinaryOp::Mul, false) => builder.ins().imul(lhs, rhs),
                                    _ => {
                                        return Err(format!(
                                            "CPU serial atomic operation {op:?} is unsupported"
                                        ));
                                    }
                                };
                                store_scalar(&mut builder, address, dtype, native);
                                if let Some(result) = result {
                                    values.insert(result, (lhs, dtype));
                                }
                            }
                            I::Assign { target, op, value } => {
                                let (address, dtype) = *places
                                    .get(target)
                                    .ok_or("CPU assignment target is not an indexed place")?;
                                let (rhs, other) = scalar_value(&values, *value)?;
                                if dtype != other {
                                    return Err("CPU assignment types differ".into());
                                }
                                let assigned = match op {
                                    seismic_lang::syntax::ast::AssignOp::Assign => rhs,
                                    operation => {
                                        let lhs = load_scalar(&mut builder, address, dtype);
                                        match (operation, cpu_type(dtype).is_float()) {
                                            (seismic_lang::syntax::ast::AssignOp::Add, true) => {
                                                builder.ins().fadd(lhs, rhs)
                                            }
                                            (seismic_lang::syntax::ast::AssignOp::Sub, true) => {
                                                builder.ins().fsub(lhs, rhs)
                                            }
                                            (seismic_lang::syntax::ast::AssignOp::Mul, true) => {
                                                builder.ins().fmul(lhs, rhs)
                                            }
                                            (seismic_lang::syntax::ast::AssignOp::Add, false) => {
                                                builder.ins().iadd(lhs, rhs)
                                            }
                                            (seismic_lang::syntax::ast::AssignOp::Sub, false) => {
                                                builder.ins().isub(lhs, rhs)
                                            }
                                            (seismic_lang::syntax::ast::AssignOp::Mul, false) => {
                                                builder.ins().imul(lhs, rhs)
                                            }
                                            _ => unreachable!(),
                                        }
                                    }
                                };
                                store_scalar(&mut builder, address, dtype, assigned);
                            }
                            I::Publish { value, destination } => {
                                let (value, dtype) = scalar_value(&values, *value)?;
                                let (address, other) = *places
                                    .get(destination)
                                    .ok_or("CPU publication destination is not an indexed place")?;
                                if dtype != other {
                                    return Err("CPU publication types differ".into());
                                }
                                store_scalar(&mut builder, address, dtype, value);
                            }
                            I::Math { op, arguments } => {
                                let args = arguments
                                    .iter()
                                    .map(|id| scalar_value(&values, *id))
                                    .collect::<Result<Vec<_>, _>>()?;
                                let dtype = args.first().ok_or("CPU math has no arguments")?.1;
                                let native = match op {
                                    seismic_lang::sir::Math::Fma => {
                                        builder.ins().fma(args[0].0, args[1].0, args[2].0)
                                    }
                                    seismic_lang::sir::Math::Sqrt => builder.ins().sqrt(args[0].0),
                                    seismic_lang::sir::Math::Abs => builder.ins().fabs(args[0].0),
                                    seismic_lang::sir::Math::Max => {
                                        builder.ins().fmax(args[0].0, args[1].0)
                                    }
                                    seismic_lang::sir::Math::Min => {
                                        builder.ins().fmin(args[0].0, args[1].0)
                                    }
                                    seismic_lang::sir::Math::Rsqrt => {
                                        let root = builder.ins().sqrt(args[0].0);
                                        let one = builder.ins().f32const(1.0);
                                        builder.ins().fdiv(one, root)
                                    }
                                    operation => {
                                        let function = match operation {
                                            seismic_lang::sir::Math::Exp => {
                                                seismic_realization::MathFunction::Exp
                                            }
                                            seismic_lang::sir::Math::ExpFast => {
                                                seismic_realization::MathFunction::ExpFast
                                            }
                                            seismic_lang::sir::Math::Log => {
                                                seismic_realization::MathFunction::Log
                                            }
                                            seismic_lang::sir::Math::Sin => {
                                                seismic_realization::MathFunction::Sin
                                            }
                                            seismic_lang::sir::Math::Cos => {
                                                seismic_realization::MathFunction::Cos
                                            }
                                            _ => unreachable!(),
                                        };
                                        let mut signature = ir::Signature::new(call_conv);
                                        signature.params.push(AbiParam::new(types::F32));
                                        signature.returns.push(AbiParam::new(types::F32));
                                        let sig = builder.import_signature(signature);
                                        let callee = builder.import_function(ir::ExtFuncData {
                                            name: ir::ExternalName::testcase(function.symbol()),
                                            signature: sig,
                                            colocated: false,
                                        });
                                        imports.push((callee, function));
                                        let call = builder.ins().call(callee, &[args[0].0]);
                                        builder.inst_results(call)[0]
                                    }
                                };
                                values.insert(
                                    result.ok_or("CPU math has no result")?,
                                    (native, dtype),
                                );
                            }
                            I::Yield(_) | I::Return { .. } => {}
                            other => {
                                return Err(format!(
                                    "CPU native instruction selection is incomplete for {other:?}"
                                ));
                            }
                        }
                    }
                }
            }
            for (header, exit, coordinate) in serial_loops.into_iter().rev() {
                let next = builder.ins().iadd_imm(coordinate, 1);
                builder.ins().jump(header, &[next.into()]);
                builder.seal_block(header);
                builder.switch_to_block(exit);
                builder.seal_block(exit);
            }
        }
    }
    let zero = builder.ins().iconst(types::I32, 0);
    builder.ins().return_(&[zero]);
    builder.finalize();
    let work_items = launch
        .geometry
        .workgroups
        .iter()
        .chain(launch.geometry.participants_per_workgroup.iter())
        .try_fold(1u64, |total, value| total.checked_mul(*value))
        .ok_or("CPU launch work-item count overflow")?;
    Ok(NativePhase {
        function,
        imports,
        work_items,
        scratch_bytes: usize::try_from(launch.kernel.resources.private_bytes)
            .map_err(|_| "CPU launch scratch exceeds usize")?,
        bindings,
    })
}

fn collect_launches(
    encoded: EncodedPlan<CpuDialect, EncodedLaunch>,
    phases: &mut Vec<NativePhase>,
) {
    for item in encoded.items {
        match item {
            EncodedScheduleItem::Phase(phase) => {
                phases.extend(phase.launches.into_iter().map(|launch| launch.native))
            }
            EncodedScheduleItem::Subplan(subplan) => collect_launches(*subplan.encoded, phases),
        }
    }
}

pub fn assemble(
    encoded: EncodedPlan<CpuDialect, EncodedLaunch>,
) -> Result<NativeExecution, String> {
    let mut storages = BTreeMap::new();
    fn collect_plan(
        plan: &seismic_realization::executable::ResolvedPlan<CpuDialect>,
        out: &mut BTreeMap<
            ResolvedStorageId,
            seismic_realization::executable::ResolvedStorage<CpuDialect>,
        >,
    ) {
        for storage in &plan.device_storage().allocations {
            out.insert(storage.id, storage.clone());
        }
        for item in plan.items().iter() {
            match item {
                ResolvedScheduleItem::Phase(phase) => {
                    for launch in phase.launches.iter() {
                        out.extend(
                            launch
                                .storage
                                .values()
                                .map(|value| (value.id, value.clone())),
                        );
                    }
                }
                ResolvedScheduleItem::Subplan(subplan) => collect_plan(&subplan.plan, out),
            }
        }
    }
    collect_plan(&encoded.resolved, &mut storages);
    let mut external = storages
        .values()
        .filter(|storage| storage.scope == StorageScope::External)
        .filter_map(|storage| storage.provenance.abi.as_ref().map(|abi| (abi, storage)))
        .collect::<Vec<_>>();
    external.sort_by_key(|(abi, _)| match abi {
        AbiRole::Parameter { ordinal, path, .. } => (0, *ordinal, path.clone()),
        AbiRole::Result { ordinal, path, .. } => (1, *ordinal, path.clone()),
        AbiRole::InvocationResource { .. } => (2, 0, Vec::new()),
    });
    let mut buffers = Vec::new();
    let mut external_ids = Vec::new();
    let mut scalar_ids = Vec::new();
    let mut scalars = Vec::new();
    let mut seen = BTreeSet::new();
    for (abi, storage) in external {
        if !seen.insert(storage.id) {
            continue;
        }
        if let (
            AbiRole::Parameter { ordinal, path, .. },
            crate::physical::CpuResolvedLayout::ScalarSlot { dtype, words },
        ) = (abi, &storage.layout)
        {
            for word in 0..*words {
                let suffix = path
                    .iter()
                    .map(u32::to_string)
                    .chain(((*words > 1).then_some(word)).map(|value| value.to_string()))
                    .collect::<Vec<_>>()
                    .join(".");
                scalars.push(seismic_lang::abi::ScalarParameter::plain(
                    if suffix.is_empty() {
                        format!("input{ordinal}")
                    } else {
                        format!("input{ordinal}.{suffix}")
                    },
                    *dtype,
                ));
                scalar_ids.push(storage.id);
            }
            continue;
        }
        let (parameter, role, plane) = match abi {
            AbiRole::Parameter {
                ordinal,
                representation_plane,
                ..
            } => (
                format!("input{ordinal}"),
                BufferRole::Parameter,
                representation_plane.clone().unwrap_or_default(),
            ),
            AbiRole::Result {
                ordinal,
                path,
                representation_plane,
            } => (
                format!("result{ordinal}"),
                BufferRole::Result { path: path.clone() },
                representation_plane.clone().unwrap_or_default(),
            ),
            AbiRole::InvocationResource { name } => {
                (name.clone(), BufferRole::Internal, String::new())
            }
        };
        buffers.push(BufferSpec {
            parameter,
            plane,
            role,
            bytes: usize::try_from(storage.bytes).map_err(|_| "CPU ABI bytes exceed usize")?,
            alignment: usize::try_from(storage.alignment)
                .map_err(|_| "CPU ABI alignment exceeds usize")?,
        });
        external_ids.push(storage.id);
    }
    let retained = storages
        .values()
        .filter(|storage| storage.scope == StorageScope::Device)
        .map(|storage| (storage.id, storage.bytes, storage.alignment))
        .collect();
    let mut phases = Vec::new();
    collect_launches(encoded, &mut phases);
    Ok(NativeExecution {
        name: "seismic_cpu_plan".into(),
        conditions: InvocationConditions::default(),
        call_conv: crate::codegen::Policy::host()?.call_conv(),
        phases,
        buffers,
        scalars,
        external_ids,
        scalar_ids,
        retained,
    })
}
