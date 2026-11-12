//! Real resident weights for tuning rotations. Each weight is imported the
//! way loading imports it, from the same source
//! ([`crate::resources::filled_import`]) through
//! the same prepared import entry, into a tensor owned by tuning; it is
//! released when its entry is tuned.

use crate::{
    ArtifactComponent, ArtifactComponentKind, ImportProgramSlot, ModelLoadPlan, Stored,
    WeightPlan,
    native::import::ImportKernels,
};
use magnitude_artifacts::{Package, gguf::GgufArtifact};
use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
use seismic::{BuildError, Device, Element, Tensor};
use std::collections::HashMap;

/// Where tuning reads weights from.
pub trait TuningWeightSource {
    /// The bytes of `weight` in its upload representation, on the host:
    /// for a weight the host itself reads or places. `leading`: as
    /// [`Self::import_source`]'s.
    fn source_bytes(&self, weight: &WeightPlan, leading: bool) -> Result<Vec<u8>, String>;

    /// The tensor an import entry reads `weight` from on `device`: its
    /// source in its upload representation, one row of every element.
    /// Tuning reads real values rather than zeros because a tuned entry's
    /// time can depend on them: an NVFP4 operand recipe quantizes the
    /// expert product row (`silu(gate) · up`), and its block scales branch
    /// on the values (`nvfp4::value`'s zero scale, `nvfp4::global_scale`'s
    /// binade corrections, `projection/stage.h`).
    fn import_source(&self, device: &Device, weight: &WeightPlan, leading: bool) -> Result<Tensor, BuildError>;
}

impl TuningWeightSource for Package {
    /// The transformed (and dequantized) source bytes, as loading uploads
    /// them; of a leading plan, those of its first experts only.
    fn source_bytes(&self, weight: &WeightPlan, leading: bool) -> Result<Vec<u8>, String> {
        let artifact = artifact(self, weight.component)?;
        let stored = Stored::of_plan(artifact, weight, leading).map_err(|error| error.to_string())?;
        let bytes = crate::programs::native_import::stored_source_bytes(&stored)
            .map_err(|error| error.to_string())?;
        crate::import_transforms::upload_bytes(weight, &stored, bytes)
    }

    /// The source as the load's own importer takes it
    /// ([`crate::resources::filled_import`]): an upload written from the
    /// stored range as it is read around the file cache. Tuning's import of
    /// a weight therefore holds on the host what the load's import of it
    /// holds, and no copy of the source besides; `leading`: the plan is of
    /// an expert stack's first experts, a prefix of the stored tensor.
    fn import_source(&self, device: &Device, weight: &WeightPlan, leading: bool) -> Result<Tensor, BuildError> {
        let artifact = artifact(self, weight.component)?;
        let stored = Stored::of_plan(artifact, weight, leading).map_err(|error| error.to_string())?;
        Ok(crate::resources::filled_import(weight, &stored, device)?)
    }
}

fn artifact(package: &Package, component: ArtifactComponent) -> Result<&GgufArtifact, String> {
    let artifact = match component.kind {
        ArtifactComponentKind::Target => package.target(),
        ArtifactComponentKind::Projector => package
            .projector()
            .ok_or("the package has no projector artifact")?,
        ArtifactComponentKind::Draft => {
            package.draft().ok_or("the package has no draft artifact")?
        }
    };
    if artifact.identity() != component.identity {
        return Err("the package artifact differs from the planned component".into());
    }
    Ok(artifact)
}

/// Zero weights of every planned shape, for programs prepared from a
/// synthetic model definition without an artifact (test fixtures). Every
/// source representation decodes zero bytes to zeros, so each entry tunes
/// on well-defined values at the model's real geometry.
pub struct ZeroTuningWeights;

impl TuningWeightSource for ZeroTuningWeights {
    fn source_bytes(&self, weight: &WeightPlan, _leading: bool) -> Result<Vec<u8>, String> {
        let bytes = weight
            .upload
            .canonical_byte_len(&[logical_count(&weight.shape)?])
            .map_err(|error| error.to_string())?;
        Ok(vec![
            0;
            usize::try_from(bytes)
                .map_err(|_| "weight bytes exceed usize")?
        ])
    }

    fn import_source(&self, device: &Device, weight: &WeightPlan, _leading: bool) -> Result<Tensor, BuildError> {
        Ok(Tensor::zeros(device, weight.upload, &[logical_count(&weight.shape)?])?)
    }

}

fn logical_count(shape: &[u64]) -> Result<u64, String> {
    crate::planning::elements(shape).ok_or_else(|| "weight element count overflows".to_owned())
}

/// The load plan's weight of one role. A banked expert stack's is its
/// resident plan: tuning measures the entries a bank shares with a resident
/// load on the weights as a resident load holds them.
pub(crate) fn weight_plan(
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
) -> Result<&WeightPlan, String> {
    let role = WeightRole { scope, kind };
    if let Some(resident) = load.expert_bank().and_then(|bank| bank.resident_plan(role)) {
        return Ok(resident);
    }
    load.planned_weights()
        .find(|weight| weight.role == role)
        .ok_or_else(|| format!("weight {role:?} is absent from the load plan"))
}

/// Whether a tensor a tuning cache keeps is bound by anything else: dropping
/// the cache's handle alone would give no storage back.
pub(crate) fn held_elsewhere(tensor: &Tensor) -> bool {
    matches!(Tensor::reclaimable_bytes([tensor]), Ok(0))
}

pub(crate) struct TuningWeights<'a> {
    device: &'a Device,
    load: &'a ModelLoadPlan,
    source: &'a dyn TuningWeightSource,
    import: &'a ImportKernels,
    resident: HashMap<Stack, Tensor>,
    /// The packing bounds the imports made (`WeightPlan::bound`), kept for
    /// the tuning's duration: a bound is a few values, and outlives its
    /// weight's release.
    bounds: HashMap<Stack, Tensor>,
}

/// One imported weight: a role, and for a banked expert stack the leading
/// experts it holds ([`TuningWeights::decode_weight`]), none of a whole one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Stack {
    role: WeightRole,
    leading: Option<u64>,
}

/// The plan of the role's stack a decode launch of `scope` reads where
/// `load`'s bank serves it: its first slots (`ExpertBankPlan::slots`), the
/// records the launch's routes address, whatever the layer's expert count.
/// None of a stack the load holds whole.
fn decode_stack(load: &ModelLoadPlan, scope: WeightScope, kind: WeightKind) -> Option<WeightPlan> {
    let bank = load.expert_bank()?;
    let plan = bank.resident_plan(WeightRole { scope, kind })?;
    (bank.slots < *plan.shape.first()?).then_some(())?;
    // A bank is its stack's first `slots` records, so the stack divides into
    // them: a stack that does not is an invariant violation, never a whole one.
    Some(plan.leading(bank.slots).expect("a banked stack divides into its bank's slots"))
}

/// The experts a decode point's routes draw from, its stacks of `scopes`
/// read as [`TuningWeights::decode_weight`] gives them: the fewest of any of
/// them, `experts` where the load holds them whole. A unit's record is of
/// the launch measured, so this is a fact of its key.
pub(crate) fn decode_experts(
    load: &ModelLoadPlan,
    scopes: &[WeightScope],
    kinds: &[WeightKind],
    experts: u64,
) -> u64 {
    scopes
        .iter()
        .flat_map(|scope| kinds.iter().filter_map(|kind| decode_stack(load, *scope, *kind)))
        .map(|stack| stack.shape[0])
        .fold(experts, u64::min)
}

/// What a decode unit's bindings add for the `read` experts of a layer's
/// `layer` its launches read ([`decode_experts`]): a record is of the
/// launch measured, so a unit at the bank's slots is not one at every
/// expert. Nothing where the load holds the stack whole, whose units keep
/// their bindings.
pub(crate) fn experts_text(read: u64, layer: u64) -> String {
    match read < layer {
        true => format!(",S={read}"),
        false => String::new(),
    }
}

impl<'a> TuningWeights<'a> {
    pub fn new(
        device: &'a Device,
        load: &'a ModelLoadPlan,
        source: &'a dyn TuningWeightSource,
        import: &'a ImportKernels,
    ) -> Self {
        Self {
            device,
            load,
            source,
            import,
            resident: HashMap::new(),
            bounds: HashMap::new(),
        }
    }

    /// Drop every imported weight that no built input binds. A weight some
    /// point's inputs still hold stays findable here, so the next point or
    /// unit that reads the same role shares its storage: one layer's weight
    /// is on the device once, however many points and units read it, and
    /// leaves when the last of them gives its inputs up.
    pub fn release_unheld(&mut self) {
        self.resident.retain(|_, tensor| held_elsewhere(tensor));
    }

    pub fn load(&self) -> &'a ModelLoadPlan {
        self.load
    }

    fn plan(&self, scope: WeightScope, kind: WeightKind) -> Result<&'a WeightPlan, String> {
        weight_plan(self.load, scope, kind)
    }

    /// The packing bound of a weight, as its import makes it; `None` for a
    /// weight without one.
    pub fn bound(
        &mut self,
        scope: WeightScope,
        kind: WeightKind,
    ) -> Result<Option<Tensor>, BuildError> {
        let role = WeightRole { scope, kind };
        if self.plan(scope, kind)?.bound == 0 {
            return Ok(None);
        }
        let stack = Stack { role, leading: None };
        if !self.bounds.contains_key(&stack) {
            self.weight(scope, kind)?;
        }
        Ok(self.bounds.get(&stack).cloned())
    }

    /// The resident weight of one role, imported once and shared while any
    /// built input holds it. An allocation the device refuses is
    /// [`BuildError::Refused`].
    pub fn weight(&mut self, scope: WeightScope, kind: WeightKind) -> Result<Tensor, BuildError> {
        let plan = self.plan(scope, kind)?;
        self.stack(Stack { role: WeightRole { scope, kind }, leading: None }, plan)
    }

    /// [`Self::weight`] of a decode launch's stack: a banked layer's, of the
    /// slots it reads ([`decode_experts`]) and not of every expert.
    pub fn decode_weight(&mut self, scope: WeightScope, kind: WeightKind) -> Result<Tensor, BuildError> {
        match decode_stack(self.load, scope, kind) {
            Some(plan) => self.stack(Stack { role: WeightRole { scope, kind }, leading: Some(plan.shape[0]) }, &plan),
            None => self.weight(scope, kind),
        }
    }

    /// The weight `stack` names, imported from `plan`: its role's, or the
    /// plan of its leading experts.
    fn stack(&mut self, stack: Stack, plan: &WeightPlan) -> Result<Tensor, BuildError> {
        if let Some(tensor) = self.resident.get(&stack) {
            return Ok(tensor.clone());
        }
        // A plane the host places is uploaded as placed, as loading
        // uploads it.
        if plan.placed_on_host() {
            let bytes = self.source.source_bytes(plan, stack.leading.is_some())?;
            let tensor = Tensor::from_host(self.device, plan.resident, &plan.shape, &bytes)?;
            self.resident.insert(stack, tensor.clone());
            return Ok(tensor);
        }
        let source = self.source.import_source(self.device, plan, stack.leading.is_some())?;
        let destination = Tensor::zeros(self.device, plan.resident, &plan.shape)?;
        let mut bound = (plan.bound != 0)
            .then(|| Tensor::zeros(self.device, Element::f32(), &[plan.bound]))
            .transpose()?;
        let handle = self
            .import
            .attested(ImportProgramSlot::of(plan))
            .ok_or_else(|| {
                format!(
                    "no import entry was prepared for {} -> {}",
                    plan.source.name(),
                    plan.resident.name()
                )
            })?;
        crate::programs::native_import::import_into(&handle, &source, &destination, bound.as_mut())
            .map_err(|error| error.to_string())?;
        if let Some(bound) = bound {
            self.bounds.insert(stack, bound);
        }
        self.resident.insert(stack, destination.clone());
        Ok(destination)
    }
}
