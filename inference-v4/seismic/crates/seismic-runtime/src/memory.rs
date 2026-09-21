//! Owner-local allocation charges. A configured limit bounds storage bytes
//! requested through this domain, not driver overhead or system-wide free RAM.
use crate::Error;
use std::{cell::Cell, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub charged: usize,
    pub limit: Option<usize>,
}
#[derive(Default)]
pub(crate) struct Domain {
    charged: Cell<usize>,
    limit: Cell<Option<usize>>,
}
impl Domain {
    pub fn usage(&self) -> Usage {
        Usage {
            charged: self.charged.get(),
            limit: self.limit.get(),
        }
    }
    pub fn set_limit(&self, limit: Option<usize>) -> Result<(), Error> {
        if let Some(limit) = limit {
            if limit < self.charged.get() {
                return Err(Error::LimitBelowCharges {
                    limit,
                    charged: self.charged.get(),
                });
            }
        }
        self.limit.set(limit);
        Ok(())
    }
    pub fn charge(self: &Rc<Self>, bytes: usize) -> Result<Charge, Error> {
        let available = self.limit.get().map_or(usize::MAX, |limit| limit) - self.charged.get();
        if bytes > available {
            return Err(Error::Capacity {
                required: bytes,
                available,
            });
        }
        self.charged.set(self.charged.get() + bytes);
        Ok(Charge {
            domain: self.clone(),
            bytes,
        })
    }
}
pub(crate) struct Charge {
    domain: Rc<Domain>,
    bytes: usize,
}
impl Charge {
    pub fn belongs_to(&self, domain: &Rc<Domain>) -> bool {
        Rc::ptr_eq(&self.domain, domain)
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.domain
            .charged
            .set(self.domain.charged.get() - self.bytes);
    }
}
