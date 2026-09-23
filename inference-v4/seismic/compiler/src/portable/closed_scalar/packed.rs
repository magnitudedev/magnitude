//! Actual packed field addresses, independent of the numerical decode recipe.
use super::*;

impl Terms {
    pub(in crate::portable) fn natural_div_rem(
        &mut self,
        remainder: bool,
        value: Term,
        divisor: u32,
    ) -> Term {
        assert_ne!(divisor, 0, "registered packet groups are nonzero");
        if let Node::Natural(value) = self.nodes[value.0] {
            return self.node(Node::Natural(if remainder {
                value % u64::from(divisor)
            } else {
                value / u64::from(divisor)
            }));
        }
        if divisor == 1 {
            return if remainder {
                self.node(Node::Natural(0))
            } else {
                value
            };
        }
        self.node(if remainder {
            Node::NaturalRem(value, divisor)
        } else {
            Node::NaturalDiv(value, divisor)
        })
    }
}

impl Analysis<'_> {
    /// Project the fixed field-read instruction through its actual view and
    /// registered packet ABI. This is address/bit transport, not a decoder.
    pub(in crate::portable) fn plane_field(
        &mut self,
        view: AnyBufferView,
        indices: &[Term],
        plane: u32,
        field: u32,
        state: &State,
    ) -> Result<Term> {
        self.coordinates_in_bounds(view, indices, state)?;
        let layout = self.storage.view(view);
        if !matches!(layout.base, ViewBase::Allocation(_))
            || !matches!(layout.mapping, seismic_ir::storage::ViewMapping::Direct | seismic_ir::storage::ViewMapping::WholeAllocation)
        {
            return Err("transformed packed place relation is unfinished");
        }
        let RepresentationKind::Packed(packet) =
            &registry::representation_info(view.representation()).kind
        else {
            return Err("field read does not address a resident packet");
        };
        let schema = packet
            .planes
            .get(plane as usize)
            .ok_or("packed field has no registered plane")?;
        if field >= schema.fields || indices.len() != layout.strides.len() {
            return Err("packed field coordinates differ from the registered ABI");
        }
        let mut logical = self.terms.node(Node::Natural(0));
        for (&index, stride) in indices.iter().zip(&layout.strides) {
            let stride = self.expression((*stride).into(), &state.slots)?;
            let displacement = self.terms.natural_binary(true, index, stride);
            logical = self.terms.natural_binary(false, logical, displacement);
        }
        let packet_index = self.terms.natural_div_rem(false, logical, packet.group);
        let packet_size = self
            .terms
            .node(Node::Natural(u64::from(packet.packet_size)));
        let packet_offset = self.terms.natural_binary(true, packet_index, packet_size);
        let offset = self.expression(layout.offset.into(), &state.slots)?;
        let base = self.terms.natural_binary(false, offset, packet_offset);
        let plane_offset = self.terms.node(Node::Natural(u64::from(schema.offset)));
        let base = self.terms.natural_binary(false, base, plane_offset);
        let local = self.terms.natural_div_rem(true, logical, packet.group);
        let group = self.terms.natural_div_rem(false, local, schema.group);
        let fields = self.terms.node(Node::Natural(u64::from(schema.fields)));
        let entry = self.terms.natural_binary(true, group, fields);
        let field = self.terms.node(Node::Natural(u64::from(field)));
        let entry = self.terms.natural_binary(false, entry, field);
        match schema.encoding {
            registry::PlaneEncoding::Dense(dtype) => {
                let width = self.terms.node(Node::Natural(u64::from(dtype.bytes())));
                let displacement = self.terms.natural_binary(true, entry, width);
                let byte = self.terms.natural_binary(false, base, displacement);
                self.read(
                    state,
                    Place {
                        root: layout.base,
                        byte,
                        dtype,
                    },
                )
            }
            registry::PlaneEncoding::Packed { .. } | registry::PlaneEncoding::FloatCode { .. } => {
                if state.writes.iter().any(|write| match (write.root(),layout.base) {
                    (ViewBase::Allocation(a),ViewBase::Allocation(b)) => self.storage.allocations_may_overlap(a,b),
                    _ => true,
                }) {
                    return Err("modified packed field relation is unfinished");
                }
                let eight = self.terms.node(Node::Natural(8));
                let base_bit = self.terms.natural_binary(true, base, eight);
                let width = self.terms.node(Node::Natural(u64::from(schema.entry_bits)));
                let entry_bit = self.terms.natural_binary(true, entry, width);
                let bit = self.terms.natural_binary(false, base_bit, entry_bit);
                Ok(self.terms.node(Node::PackedBits {
                    root: layout.base,
                    bit,
                    width: schema.entry_bits,
                }))
            }
        }
    }
}
