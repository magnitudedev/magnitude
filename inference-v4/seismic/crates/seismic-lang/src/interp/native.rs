//! Target-dependent forms under a single participant: intrinsics and packed plane accessors.
use super::exec::Frame;
use super::scalar;
use super::value::{Backing, Frag, Shaped, Value};
use super::{Interpreter, TensorData};
use crate::intrinsics::Operation;
use crate::numeric::f16_to_f32;
use crate::repr::{self, Coefficient, Plane, PlaneEncoding, Repr};
use crate::sir::{Expr, Math};
use crate::types::Ty;
use crate::types::{DType, Elem};
use std::cell::RefCell;
use std::rc::Rc;

/// One decoded entry of a physical plane.
fn plane_value(rep: &Repr, planes: &[Vec<u8>], plane: &Plane, entry: usize) -> Result<f64, String> {
    let bytes = rep.plane_index(plane.name).and_then(|i| planes.get(i)).ok_or_else(|| format!("`{}` has no plane `{}`", rep.name, plane.name))?;
    let outside = || format!("plane `{}` entry {entry} is outside its storage", plane.name);
    match &plane.encoding {
        PlaneEncoding::Packed { bits, interpretation } => {
            if (entry + 1) * *bits as usize > bytes.len() * 8 {
                return Err(outside());
            }
            Ok(f64::from(interpretation.decode(repr::read_packed(bytes, entry, *bits), *bits)))
        }
        PlaneEncoding::Dense(dtype) => {
            let width = dtype.bytes() as usize;
            let raw = bytes.get(entry * width..(entry + 1) * width).ok_or_else(outside)?;
            Ok(f64::from(match (dtype, raw) {
                (DType::F32, [a, b, c, d]) => f32::from_le_bytes([*a, *b, *c, *d]),
                (DType::F16, [a, b]) => f16_to_f32(u16::from_le_bytes([*a, *b])),
                (DType::BF16, [a, b]) => f32::from_bits(u32::from(u16::from_le_bytes([*a, *b])) << 16),
                _ => return Err(format!("plane `{}` is not a floating coefficient plane", plane.name)),
            }))
        }
    }
}

/// The logical scale or bias applying to the value at flat position `flat`.
fn coefficient(rep: &Repr, planes: &[Vec<u8>], bias: bool, flat: usize) -> Result<f64, String> {
    match rep.coefficient(bias) {
        None => Err(format!("`{}` has no bias", rep.name)),
        Some(Coefficient::Direct { plane }) => plane_value(rep, planes, &plane, flat / plane.group as usize),
        Some(Coefficient::Product { factor, coefficients, field, sign }) => {
            let code = plane_value(rep, planes, &coefficients, flat / coefficients.group as usize * coefficients.fields as usize + field as usize)?;
            let factor = plane_value(rep, planes, &factor, flat / factor.group as usize)?;
            Ok(f64::from((factor as f32 * code as f32) * sign as f32))
        }
    }
}

impl<'a> Interpreter<'a> {
    /// `t.words`, `t.scale`, `t.bias` and the other physical planes of a packed tile or view:
    /// a dense tile whose packet axis is replaced by the plane's extent over that axis.
    pub(super) fn accessor(&self, s: &Shaped, axis: usize, name: &str) -> Result<Value, String> {
        let Backing::Tensor(id) = &s.backing else { return Err(format!("`.{name}` needs a packed value")) };
        let TensorData::Packed { repr: rep, planes, .. } = &self.tensors[*id] else { return Err(format!("`.{name}` needs a packed value")) };
        if axis >= s.shape.len() || s.strides[axis] != 1 {
            return Err(format!("`.{name}` needs the packet axis in storage order"));
        }
        let length = s.shape[axis];
        let logical = name == "scale" || name == "bias";
        let plane = if logical { None } else { Some(rep.plane(name).ok_or_else(|| format!("`{}` has no physical plane `{name}`", rep.name))?) };
        let group = plane.as_ref().map_or(rep.group, |p| p.group) as usize;
        let (count, dtype) = match &plane {
            None => (length.div_ceil(group), rep.coefficient_dtype()),
            Some(p) => (p.storage_elements(length as u64).and_then(|n| usize::try_from(n).ok()).ok_or("plane extent overflow")?, p.dtype()),
        };
        let mut rows = s.clone();
        rows.shape[axis] = 1;
        let mut data = Vec::with_capacity(rows.count() * count);
        // Rows are visited in row-major order of the result, whose packet axis is `axis`.
        let outer: usize = s.shape[..axis].iter().product();
        let inner: usize = s.shape[axis + 1..].iter().product();
        let starts = rows.flats();
        for o in 0..outer {
            for e in 0..count {
                for i in 0..inner {
                    let first = starts[o * inner + i];
                    if first % group != 0 {
                        return Err(format!("`.{name}` of a selection that does not start on a group of {group}"));
                    }
                    data.push(match &plane {
                        None => coefficient(rep, planes, name == "bias", first + e * group)?,
                        Some(p) => {
                            let entry = first / group * p.fields as usize;
                            match &p.encoding {
                                PlaneEncoding::Dense(_) => plane_value(rep, planes, p, entry + e)?,
                                PlaneEncoding::Packed { bits, .. } => {
                                    let bit = entry * *bits as usize;
                                    if bit % 32 != 0 {
                                        return Err(format!("`.{name}` of a selection that does not start on a storage word"));
                                    }
                                    let bytes = rep.plane_index(p.name).and_then(|k| planes.get(k)).ok_or("missing plane storage")?;
                                    let at = (bit / 32 + e) * 4;
                                    let word: [u8; 4] = std::array::from_fn(|k| bytes.get(at + k).copied().unwrap_or(0));
                                    f64::from(u32::from_le_bytes(word))
                                }
                            }
                        }
                    });
                }
            }
        }
        let mut shape = s.shape.clone();
        shape[axis] = count;
        Ok(Value::Tile(Shaped::owned(dtype, shape, data)))
    }

    fn fragment(&mut self, e: &'a Expr, f: &mut Frame<'a>) -> Result<Rc<RefCell<Frag>>, String> {
        match self.expr(e, f)? {
            Value::Native(frag) => Ok(frag),
            other => Err(format!("{} where a matrix fragment is required", other.kind())),
        }
    }

    /// The 8x8 block of a rank-2 tile or view at tile-relative `(row, column)`.
    fn block8(&mut self, tile: &'a Expr, row: &'a Expr, column: &'a Expr, f: &mut Frame<'a>) -> Result<(Shaped, Vec<usize>), String> {
        let s = self.place(tile, f)?;
        let (r, c) = (self.int(row, f)?, self.int(column, f)?);
        if s.shape.len() != 2 {
            return Err("matrix fragments address rank-2 tiles".into());
        }
        let inside = |at: i64, extent: usize| usize::try_from(at).ok().filter(|p| p + 8 <= extent);
        let (Some(r), Some(c)) = (inside(r, s.shape[0]), inside(c, s.shape[1])) else {
            return Err(format!("8x8 fragment at ({r}, {c}) outside a tile of shape {:?}", s.shape));
        };
        let flats = (0..64).map(|k| s.offset + (r + k / 8) * s.strides[0] + (c + k % 8) * s.strides[1]).collect();
        Ok((s, flats))
    }

    pub(super) fn intrinsic(&mut self, op: Operation, args: &'a [Expr], e: &'a Expr, f: &mut Frame<'a>) -> Result<Value, String> {
        let arity = op.signature().params.len();
        if args.len() != arity {
            return Err(format!("{op} takes {arity} arguments"));
        }
        match op {
            // One participant: its index is zero, an exchange returns its own value, and a
            // subgroup reduction of one value is that value.
            Operation::LaneIndex => Ok(Value::int(0)),
            Operation::ShuffleIndex => {
                let v = self.expr(&args[0], f)?;
                match self.int(&args[1], f)? {
                    0 => Ok(v),
                    other => Err(format!("participant {other} outside a group of one")),
                }
            }
            Operation::SimdSum | Operation::SimdMax | Operation::SimdMin => self.expr(&args[0], f),
            Operation::Matrix => {
                let dtype = match &e.ty {
                    Ty::Native(n) => match n.elem.as_ref().map(|elem| f.elem(elem)) {
                        Some(Elem::Dtype(d)) => d,
                        _ => DType::F32,
                    },
                    _ => DType::F32,
                };
                Ok(Value::Native(Rc::new(RefCell::new(Frag { dtype, data: [0.0; 64] }))))
            }
            Operation::MatrixLoad | Operation::MatrixLoadTranspose => {
                let frag = self.fragment(&args[0], f)?;
                let (s, flats) = self.block8(&args[1], &args[2], &args[3], f)?;
                let from = self.dtype_of(&s);
                let mut frag = frag.borrow_mut();
                for (k, flat) in flats.into_iter().enumerate() {
                    let at = if op == Operation::MatrixLoadTranspose { k % 8 * 8 + k / 8 } else { k };
                    frag.data[at] = scalar::cast(frag.dtype, (from, self.read_flat(&s, flat)?)).1;
                }
                Ok(Value::Void)
            }
            Operation::MatrixStore => {
                let frag = self.fragment(&args[0], f)?;
                let (s, flats) = self.block8(&args[1], &args[2], &args[3], f)?;
                let (dtype, data) = {
                    let frag = frag.borrow();
                    (frag.dtype, frag.data)
                };
                for (k, flat) in flats.into_iter().enumerate() {
                    self.write_flat(&s, flat, (dtype, data[k]))?;
                }
                Ok(Value::Void)
            }
            Operation::MatrixMultiplyAccumulate => {
                let dst = self.fragment(&args[0], f)?;
                let a = self.fragment(&args[1], f)?;
                let b = self.fragment(&args[2], f)?;
                let c = self.fragment(&args[3], f)?;
                let (a, b, c) = ((a.borrow().dtype, a.borrow().data), (b.borrow().dtype, b.borrow().data), (c.borrow().dtype, c.borrow().data));
                let mut dst = dst.borrow_mut();
                // The reference order: each output continues its fma chain over ascending k.
                for i in 0..8 {
                    for j in 0..8 {
                        let mut s = scalar::cast(dst.dtype, (c.0, c.1[i * 8 + j]));
                        for k in 0..8 {
                            s = scalar::cast(dst.dtype, scalar::math(Math::Fma, &[(a.0, a.1[i * 8 + k]), (b.0, b.1[k * 8 + j]), s])?);
                        }
                        dst.data[i * 8 + j] = s.1;
                    }
                }
                Ok(Value::Void)
            }
        }
    }
}
