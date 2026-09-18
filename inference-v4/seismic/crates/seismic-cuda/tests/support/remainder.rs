use cranelift_codegen::{
    cursor::{Cursor, FuncCursor},
    ir::{self, AbiParam, InstBuilder, MemFlags, Type, types},
    isa::CallConv,
};

/// Direct scalar-SSA contract fixture: language `%` has a different numeric
/// contract and source validity guards, so it cannot test every Srem input.
pub fn program(ty: Type, elements: usize) -> seismic_realization::ScalarProgram {
    let mut function = ir::Function::new();
    function.signature = ir::Signature::new(CallConv::SystemV);
    function.signature.params = vec![AbiParam::new(types::I64); 3];
    function.signature.returns = vec![AbiParam::new(types::I32)];
    let block = function.dfg.make_block();
    function.layout.append_block(block);
    let table = function.dfg.append_block_param(block, types::I64);
    function.dfg.append_block_param(block, types::I64);
    function.dfg.append_block_param(block, types::I64);
    let mut cursor = FuncCursor::new(&mut function);
    cursor.goto_bottom(block);
    let a = cursor.ins().load(types::I64, MemFlags::new(), table, 0);
    let b = cursor.ins().load(types::I64, MemFlags::new(), table, 8);
    let out = cursor.ins().load(types::I64, MemFlags::new(), table, 16);
    for i in 0..elements {
        let offset = (i * ty.bytes() as usize) as i32;
        let lhs = cursor.ins().load(ty, MemFlags::new(), a, offset);
        let rhs = cursor.ins().load(ty, MemFlags::new(), b, offset);
        let value = cursor.ins().srem(lhs, rhs);
        cursor.ins().store(MemFlags::new(), value, out, offset);
    }
    let success = cursor.ins().iconst(types::I32, 0);
    cursor.ins().return_(&[success]);
    seismic_realization::ScalarProgram {
            conditions: Default::default(),
        function,
        buffers: ["a", "b", "out"]
            .into_iter()
            .map(|parameter| seismic_realization::BufferSpec {
                parameter: parameter.into(),
                plane: String::new(),
                bytes: elements * ty.bytes() as usize,
                alignment: ty.bytes() as usize,
            })
            .collect(),
        scalars: vec![],
        scratch_bytes: 0,
        imports: vec![],
        backend_calls: vec![], participation: seismic_realization::dispatch::Participation::Thread,
        work_items: 1,
        dispatch: seismic_realization::Dispatch::Sequential,
        loads: vec![],
        execution: Default::default(),
    }
}

pub fn operands(ty: Type) -> Vec<(i64, i64)> {
    let minimum = (-(1i128 << (ty.bits() - 1))) as i64;
    vec![
        (minimum, -1),
        (minimum, 3),
        (minimum, minimum),
        (-7, 3),
        (7, -3),
        (-7, -3),
        (7, 3),
        (0, -3),
        (minimum + 1, minimum),
        (1, minimum),
    ]
}

#[allow(dead_code)] // Shared fixture: encoding is needed by native tests only.
pub fn bytes(values: impl IntoIterator<Item = i64>, ty: Type) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| v.to_le_bytes()[..ty.bytes() as usize].to_vec())
        .collect()
}
