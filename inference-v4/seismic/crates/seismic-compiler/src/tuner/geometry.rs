//! Unresolved dispatch geometry for the independent solver. All equations come
//! from realization; this is neither an execution family nor a timing objective.
//! Backend topology, applicability and operation accounting must be composed with
//! these constraints before a solver result can authorize a TunedIr.
use magnitude_solver::model::{Domain, ModelBuilder};
use seismic_realization::dispatch::{
    GroupDispatch, TileDeclaration, TileLayout, WorkMapping, geometry,
};

pub use seismic_accounting::algebra::Error;
use seismic_accounting::algebra::{Algebra, Symbolic, Value};
use seismic_lang::types::DType;

/// Full geometry boundary, not merely a work count. Steps remain distinct even
/// when their quotients agree, because their emitted coordinates/tails differ.
pub struct Geometry {
    pub steps: Vec<Value>,
    pub counts: Vec<Value>,
    pub strides: Vec<Value>,
    pub work_items: Value,
    pub lanes_per_item: Value,
    pub items_per_group: Value,
    pub groups: Value,
    pub threads_per_group: Value,
    pub dispatched_lanes: Value,
    pub storage: Vec<Storage>,
    extents: Vec<u64>,
    tiles: Vec<Tile>,
}
/// A tile whose capacity is still a source-family expression. Its value is the
/// same variable used by stream/packet choices, not a second independent choice.
#[derive(Clone)]
pub struct Tile {
    pub symbol: String,
    pub dtype: DType,
    pub capacity: Value,
    pub placement: seismic_realization::dispatch::TilePlacement,
}
pub struct Storage {
    pub private_elements_per_lane: Value,
    pub shared_elements_per_item: Value,
    pub private_bytes_per_lane: Value,
    pub shared_bytes_per_group: Value,
}
pub struct Reconstructed {
    pub mapping: WorkMapping,
    pub dispatch: GroupDispatch,
    pub storage: Vec<TileLayout>,
    pub tiles: Vec<TileDeclaration>,
}
impl Geometry {
    /// Domains are supplied by the backend's admitted choice definitions. No
    /// candidates are sampled, enumerated, scored or removed here. On error the
    /// caller must discard the partially constructed builder.
    pub fn append(
        builder: &mut ModelBuilder,
        name: &str,
        extents: &[u64],
        steps: &[Domain],
        lanes: Domain,
        items: Domain,
        tiles: &[TileDeclaration],
    ) -> Result<Self, Error> {
        if extents.len() != steps.len() {
            return Err(Error::Invalid("one step domain required per axis".into()));
        }
        let mut a = Symbolic::new(builder, name);
        let steps = steps
            .iter()
            .cloned()
            .map(|d| a.positive("step", d))
            .collect::<Result<Vec<_>, _>>()?;
        let lanes = a.positive("lanes", lanes)?;
        let items = a.positive("items", items)?;
        let tiles = tiles
            .iter()
            .map(|tile| {
                Ok(Tile {
                    symbol: tile.symbol.clone(),
                    dtype: tile.dtype,
                    capacity: a.constant(tile.capacity)?,
                    placement: tile.placement.clone(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Self::from_values(builder, name, extents, &steps, lanes, items, &tiles)
    }

    /// Attach the actual source/backend variables to their dispatch and storage
    /// consequences without introducing duplicate choice variables. Caller owns
    /// the complete activation guard; all inputs must belong to this builder.
    pub fn from_values(
        builder: &mut ModelBuilder,
        name: &str,
        extents: &[u64],
        steps: &[Value],
        lanes: Value,
        items: Value,
        tiles: &[Tile],
    ) -> Result<Self, Error> {
        if extents.len() != steps.len() {
            return Err(Error::Invalid("one step value required per axis".into()));
        }
        if steps
            .iter()
            .chain([&lanes, &items])
            .any(|v| v.bounds().0 == 0)
        {
            return Err(Error::Invalid(
                "dispatch steps and widths must be positive".into(),
            ));
        }
        let mut a = Symbolic::new(builder, name);
        let mapping = geometry::mapping(&mut a, extents, steps)?;
        let dispatch = geometry::dispatch(&mut a, mapping.work_items, lanes, items)?;
        let storage = tiles
            .iter()
            .map(|tile| {
                let s = geometry::storage(
                    &mut a,
                    tile.capacity,
                    u64::from(tile.dtype.bytes()),
                    &tile.placement,
                    lanes,
                    items,
                )?;
                Ok(Storage {
                    private_elements_per_lane: s.private_elements_per_lane,
                    shared_elements_per_item: s.shared_elements_per_item,
                    private_bytes_per_lane: s.private_bytes_per_lane,
                    shared_bytes_per_group: s.shared_bytes_per_group,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            steps: steps.to_vec(),
            counts: mapping.counts,
            strides: mapping.strides,
            work_items: mapping.work_items,
            lanes_per_item: lanes,
            items_per_group: items,
            groups: dispatch.groups,
            threads_per_group: dispatch.threads_per_group,
            dispatched_lanes: dispatch.dispatched_lanes,
            storage,
            extents: extents.to_vec(),
            tiles: tiles.to_vec(),
        })
    }

    /// Checks every derived field against concrete realization. This validates
    /// geometry only: the enclosing family must validate model identity, domain
    /// membership, coverage, objective and backend reconstruction independently.
    pub fn reconstruct(&self, values: &[i64]) -> Result<Reconstructed, Error> {
        let read = |id: Value| {
            values
                .get(id.id().0)
                .and_then(|v| u64::try_from(*v).ok())
                .ok_or_else(|| Error::Reconstruction("missing or negative geometry value".into()))
        };
        let check = |id: Value, expected: u64| -> Result<(), Error> {
            if read(id)? != expected {
                return Err(Error::Reconstruction(format!(
                    "geometry variable {} differs from realization",
                    id.id().0
                )));
            }
            Ok(())
        };
        let steps = self
            .steps
            .iter()
            .map(|id| read(*id))
            .collect::<Result<Vec<_>, _>>()?;
        let mapping = WorkMapping::new(&self.extents, &steps).map_err(Error::Reconstruction)?;
        for (index, axis) in mapping.axes().iter().enumerate() {
            check(self.counts[index], axis.extent)?;
            check(self.strides[index], axis.stride)?;
        }
        check(self.work_items, mapping.work_items())?;
        let dispatch = GroupDispatch::new(
            mapping.work_items(),
            read(self.lanes_per_item)?,
            read(self.items_per_group)?,
        )
        .map_err(Error::Reconstruction)?;
        check(self.groups, dispatch.groups)?;
        check(self.threads_per_group, dispatch.threads_per_group)?;
        check(self.dispatched_lanes, dispatch.dispatched_lanes())?;
        let mut storage = Vec::new();
        let mut tiles = Vec::new();
        for (tile, vars) in self.tiles.iter().zip(&self.storage) {
            let tile = TileDeclaration {
                symbol: tile.symbol.clone(),
                dtype: tile.dtype,
                capacity: read(tile.capacity)?,
                placement: tile.placement.clone(),
            };
            let layout = tile.layout(&dispatch).map_err(Error::Reconstruction)?;
            check(
                vars.private_elements_per_lane,
                layout.private_elements_per_lane,
            )?;
            check(
                vars.shared_elements_per_item,
                layout.shared_elements_per_item,
            )?;
            check(vars.private_bytes_per_lane, layout.private_bytes_per_lane)?;
            check(vars.shared_bytes_per_group, layout.shared_bytes_per_group)?;
            storage.push(layout);
            tiles.push(tile);
        }
        Ok(Reconstructed {
            mapping,
            dispatch,
            storage,
            tiles,
        })
    }
}
