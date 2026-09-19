//! Joint launch grouping with symbolic padding and storage. This exports the
//! unresolved geometry of the original retained family, without a candidate table.
use super::{Execution, GroupFamily};
use magnitude_solver::model::{Constraint, Domain, LinearTerm, ModelBuilder};
use seismic_compiler::tuner::geometry::{Error, Geometry};
use std::sync::Arc;

pub struct DispatchFamily {
    pub launches: Vec<Geometry>,
    owner: Arc<GroupFamily>,
    selected: Vec<u64>,
}
impl GroupFamily {
    /// Every launch has its original independent interval. Padding constraints
    /// are mathematical relations rather than a pre-enumerated list of members.
    pub fn append_dispatch(
        self: &Arc<Self>,
        builder: &mut ModelBuilder,
        name: &str,
        selected: &[u64],
    ) -> Result<DispatchFamily, Error> {
        if selected.len() > self.launches.len() {
            return Err(Error::Invalid("too many selected Metal launches".into()));
        }
        let integer = |n| {
            i64::try_from(n)
                .map_err(|_| Error::Unsupported("Metal dispatch domain exceeds i64".into()))
        };
        let mut launches = Vec::with_capacity(self.launches.len());
        for (index, dispatch) in self.launches.iter().enumerate() {
            let items = if let Some(&items) = selected.get(index) {
                if !self.admits_grouping(index, items).map_err(Error::Invalid)? {
                    return Err(Error::Invalid(
                        "selected Metal grouping is outside its original domain".into(),
                    ));
                }
                Domain::singleton(integer(items)?)
            } else {
                Domain::interval(1, integer(self.maximum_by_launch[index])?)
                    .map_err(|e| Error::Invalid(e.to_string()))?
            };
            let geometry = Geometry::append(
                builder,
                &format!("{name}.launch{index}"),
                &[dispatch.work_items],
                &[Domain::singleton(1)],
                Domain::singleton(integer(dispatch.lanes_per_item)?),
                items,
                &self.execution.memory().launches()[index].slots,
            )?;
            builder.constraint(Constraint::LinearLe {
                terms: vec![LinearTerm::new(geometry.dispatched_lanes.id(), 1)],
                rhs: i128::from(u32::MAX) * i128::from(dispatch.lanes_per_item),
            });
            launches.push(geometry);
        }
        Ok(DispatchFamily {
            launches,
            owner: self.clone(),
            selected: selected.to_vec(),
        })
    }
}
impl DispatchFamily {
    pub fn reconstruct(&self, values: &[i64]) -> Result<Execution, Error> {
        let expected = self
            .launches
            .iter()
            .map(|g| g.reconstruct(values))
            .collect::<Result<Vec<_>, _>>()?;
        let items: Vec<_> = expected
            .iter()
            .map(|g| g.dispatch.items_per_group)
            .collect();
        if !items.starts_with(&self.selected) {
            return Err(Error::Reconstruction(
                "Metal dispatch changed a previously selected launch".into(),
            ));
        }
        let execution = self
            .owner
            .select_launches(&items)
            .map_err(Error::Reconstruction)?;
        let actual = execution
            .phases()
            .iter()
            .flat_map(|p| std::iter::once(&p.dispatch).chain(p.merge_dispatch.as_ref()));
        for (actual, expected) in actual.zip(expected) {
            if *actual != expected.dispatch {
                return Err(Error::Reconstruction(
                    "Metal dispatch differs from its symbolic family".into(),
                ));
            }
        }
        Ok(execution)
    }
}
