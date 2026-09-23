//! Source continuation rendezvous, constructed from ordinary kernel storage,
//! control, and synchronization operations before closure.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scope { Subgroup, Workgroup }
impl Scope {
    pub(super) fn uniformity(self) -> registry::IntrinsicUniformity {
        match self {
            Self::Subgroup => registry::IntrinsicUniformity::Subgroup,
            Self::Workgroup => registry::IntrinsicUniformity::Workgroup,
        }
    }
    fn barrier<B: seismic_target::TargetFamily>(self, kernel: &mut PortableBuilder<'_, B>) {
        match self {
            Self::Subgroup => kernel.subgroup_barrier(),
            Self::Workgroup => kernel.workgroup_barrier(),
        }
    }
    pub(super) fn combine(self, other: Self) -> Self {
        if self == Self::Workgroup || other == Self::Workgroup { Self::Workgroup } else { Self::Subgroup }
    }
    pub(super) fn region(function:&SemanticFunction, region:RegionId,
        helpers:&BTreeMap<FamilyId,&SemanticFunction>) -> Option<Self> {
        function.nodes(region).filter_map(|(id,_)|Self::node(function,id,helpers)).reduce(Self::combine)
    }
    pub(super) fn node(function:&SemanticFunction, id:NodeId,
        helpers:&BTreeMap<FamilyId,&SemanticFunction>) -> Option<Self> {
        match function.node(id).view() {
            SemanticNodeView::Intrinsic { intrinsic, .. } => match registry::intrinsic_signature(intrinsic).effects.participation {
                registry::IntrinsicParticipation::Independent => None,
                registry::IntrinsicParticipation::FullSubgroup => Some(Self::Subgroup),
                registry::IntrinsicParticipation::FullWorkgroup => Some(Self::Workgroup),
            },
            SemanticNodeView::Call { family, .. } => {
                let child = helpers[&family];
                Self::region(child,child.root(),helpers)
            }
            SemanticNodeView::Loop { body, .. } => Self::region(function,body,helpers),
            SemanticNodeView::If { then, otherwise, .. } =>
                [Self::region(function,then,helpers),Self::region(function,otherwise,helpers)]
                    .into_iter().flatten().reduce(Self::combine),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(super) struct Cohort {
    live: PortableTensor,
    participants: NatExpr,
    lane: PortableValue,
    count: PortableValue,
}
impl Cohort {
    pub(super) fn new<B: seismic_target::TargetFamily>(
        kernel: &mut PortableBuilder<'_, B>, domain: SegmentLaunchDomain,
    ) -> Self {
        let participants = kernel.expression_arena().nat_product(&domain.workgroup);
        let live = kernel.local_tensor(LaunchLocalKind::Workgroup,
            registry::dense(DType::U32), vec![participants]);
        let x = kernel.local_id(0);
        let y = kernel.local_id(1);
        let z = kernel.local_id(2);
        let sx = kernel.workgroup_size(0);
        let sy = kernel.workgroup_size(1);
        let zy = kernel.binary(BinaryOp::Mul, z, sy);
        let yz = kernel.binary(BinaryOp::Add, y, zy);
        let rows = kernel.binary(BinaryOp::Mul, yz, sx);
        let lane = kernel.binary(BinaryOp::Add, x, rows);
        let count = kernel.nat_arg(participants);
        Self { live, participants, lane, count }
    }

    /// Admission padding contains whole cohorts. It is not a stopped source
    /// participant and never writes the live table for an admitted cohort.
    pub(super) fn membership<B: seismic_target::TargetFamily>(
        &self, kernel:&mut PortableBuilder<'_,B>, scope:Scope,
        extent:NatExpr, logical_base:Option<seismic_ir::kernel::dynamic::LogicalIndexBinding>,
    ) -> PortableValue {
        let group = kernel.workgroup_id(0);
        let group_start = kernel.binary(BinaryOp::Mul,group,self.count);
        let cohort_start = match scope {
            Scope::Workgroup => group_start,
            Scope::Subgroup => {
                let ordinal = kernel.subgroup_ordinal();
                let width = kernel.subgroup_size();
                let offset = kernel.binary(BinaryOp::Mul,ordinal,width);
                kernel.binary(BinaryOp::Add,group_start,offset)
            }
        };
        let extent = kernel.nat_arg(extent);
        let remaining = if let Some(binding) = logical_base {
            let base = kernel.logical_base(binding);
            kernel.binary(BinaryOp::Sub,extent,base)
        } else { extent };
        kernel.cmp(CmpOp::Lt,cohort_start,remaining)
    }

    /// All members execute both barriers, including stopped source members.
    /// `values` are actual source results whose successful equality has already
    /// been derived by the source value owner. This is a private construction
    /// helper, not a caller-set uniformity operation.
    pub(super) fn gate<B: seismic_target::TargetFamily>(
        &self, kernel: &mut PortableBuilder<'_, B>, scope: Scope,
        alive: PortableValue, values: &[PortableValue],
    ) -> (PortableValue, Vec<PortableValue>) {
        let live = kernel.scalar_bits(alive);
        kernel.tensor_write(&self.live, &[self.lane], live);
        let zero = kernel.index_constant(0);
        let (first, width) = match scope {
            Scope::Workgroup => (zero, self.count),
            Scope::Subgroup => {
                let ordinal = kernel.subgroup_ordinal();
                let width = kernel.subgroup_size();
                (kernel.binary(BinaryOp::Mul, ordinal, width), width)
            }
        };
        let mut tables = Vec::with_capacity(values.len());
        let radix = kernel.index_constant(1u64 << 32);
        for value in values {
            let words = if value.ty() == ValueType::Index {
                let low = kernel.cast(*value, ValueType::Scalar(DType::U32));
                let high = kernel.binary(BinaryOp::Div, *value, radix);
                vec![low, kernel.cast(high, ValueType::Scalar(DType::U32))]
            } else { vec![kernel.scalar_bits(*value)] };
            let mut storage = Vec::new();
            for word in words {
                let table = kernel.local_tensor(LaunchLocalKind::Workgroup,
                    registry::dense(DType::U32), vec![self.participants]);
                kernel.tensor_write(&table, &[self.lane], word);
                storage.push(table);
            }
            tables.push(storage);
        }
        scope.barrier(kernel);
        let end = kernel.binary(BinaryOp::Add, first, width);
        let yes = kernel.constant(ConstantValue::Bool(true), ValueType::Bool);
        let all_live = kernel.repeat(first, end, vec![yes], &[scope.uniformity()], |body, index, carry| {
            let word = body.tensor_read(&self.live, &[index]);
            let live = body.scalar_from_bits(word, DType::Bool);
            vec![body.logic(LogicOp::And, carry[0], live)]
        })[0];
        let promoted = kernel.branch(all_live, |body| {
            values.iter().zip(&tables).map(|(value, words)| {
                let low = body.tensor_read(&words[0], &[first]);
                match value.ty() {
                    ValueType::Index => {
                        let high = body.tensor_read(&words[1], &[first]);
                        let low = body.cast(low, ValueType::Index);
                        let high = body.cast(high, ValueType::Index);
                        let upper = body.binary(BinaryOp::Mul, high, radix);
                        body.binary(BinaryOp::Add, upper, low)
                    }
                    ValueType::Bool => body.scalar_from_bits(low, DType::Bool),
                    ValueType::Scalar(dtype) => body.scalar_from_bits(low, dtype),
                    _ => unreachable!("cohort transport is a scalar result"),
                }
            }).collect()
        }, |body| values.iter().map(|v| zero_of(body, v.ty())).collect());
        // Protect both the live scan and representative payload reads from
        // every member's next reuse of the same cells.
        scope.barrier(kernel);
        (all_live, promoted)
    }
}
