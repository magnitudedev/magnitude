use super::*;

#[derive(Clone)]
pub struct Pending {
    workflow: usize,
    node: usize,
    leaf: usize,
    ty: SignatureType,
    slice: Option<(u64, u64)>,
}
impl Pending {
    pub fn signature(&self) -> &SignatureType {
        &self.ty
    }
    pub fn slice(&self, start: u64, end: u64) -> Result<Self, Error> {
        if !matches!(self.ty,SignatureType::Tensor{rank,..} if rank>0)
            || end < start
            || self.slice.is_some()
        {
            return Err(Error::new("TypeError","pending slices require a rank-positive tensor and explicit nonnegative bounds; nested slices are unsupported"));
        }
        let mut result = self.clone();
        result.slice = Some((start, end));
        Ok(result)
    }
}
#[derive(Clone)]
pub enum WorkflowValue {
    External(Value),
    Pending(Pending),
    Move(Pending),
    Tuple(Vec<WorkflowValue>),
    Unit,
}
struct Node {
    kernel: Arc<Kernel>,
    args: Vec<WorkflowValue>,
}
pub struct Workflow {
    id: usize,
    device: Device,
    nodes: Vec<Node>,
    closed: bool,
    moved_external: BTreeSet<usize>,
    moved_pending: BTreeSet<(usize, usize)>,
}
impl Workflow {
    pub fn new(device: &Device) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            device: device.clone(),
            nodes: Vec::new(),
            closed: false,
            moved_external: BTreeSet::new(),
            moved_pending: BTreeSet::new(),
        }
    }
    pub fn close(&mut self) {
        self.closed = true;
        self.nodes.clear();
    }
    pub fn enqueue(
        &mut self,
        kernel: Arc<Kernel>,
        args: Vec<WorkflowValue>,
    ) -> Result<WorkflowValue, Error> {
        if self.closed {
            return Err(Error::new("WorkflowError", "workflow is closed"));
        }
        if kernel.is_native() || !kernel.device.same_device(&self.device) {
            return Err(Error::new(
                "WorkflowError",
                "workflow requires ordinary kernels on its opened device",
            ));
        }
        if args.len() != kernel.function.parameters().len() {
            return Err(Error::new("TypeError", "wrong argument count"));
        }
        let mut external = self.moved_external.clone();
        let mut pending = self.moved_pending.clone();
        for ((_, ty), v) in kernel.function.parameters().iter().zip(&args) {
            validate(self, ty, v, &kernel.elements, &mut external, &mut pending)?;
        }
        let node = self.nodes.len();
        let mut leaf = 0;
        let result = pending_tree(
            kernel.function.result_type(),
            &kernel.elements,
            self.id,
            node,
            &mut leaf,
        );
        self.nodes.push(Node { kernel, args });
        self.moved_external = external;
        self.moved_pending = pending;
        Ok(result)
    }
    /// Single use even when admission fails. No external moves happen until
    /// the runtime has admitted the entire graph.
    pub fn run(&mut self, outputs: WorkflowValue) -> Result<WorkflowValue, Error> {
        if std::mem::replace(&mut self.closed, true) {
            return Err(Error::new("WorkflowError", "workflow is closed"));
        }
        let nodes = std::mem::take(&mut self.nodes);
        validate_outputs(self, &outputs)?;
        let mut tensors = Vec::new();
        for node in &nodes {
            for v in &node.args {
                collect(v, &mut tensors);
            }
        }
        let mut storages: Vec<_> = tensors.iter().map(|t| t.0.storage.clone()).collect();
        storages.sort_by_key(|s| Arc::as_ptr(s) as usize);
        storages.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let _guards: Vec<_> = storages.iter().map(|s| lock(&s.gate)).collect();
        let mut draft = runtime::workflow(self.device.inner());
        let mut refs: Vec<Vec<runtime::WorkflowResultRef>> = Vec::new();
        let mut moves = Vec::new();
        for node in &nodes {
            let mut args = runtime::EncodedWorkflowArgs::new();
            for ((_, ty), v) in node.kernel.function.parameters().iter().zip(&node.args) {
                encode_workflow(ty, v, &node.kernel.elements, &refs, &mut args, &mut moves)?;
            }
            let KernelKind::Ordinary(kernel) = &node.kernel.inner else {
                unreachable!()
            };
            let mut results = runtime::enqueue(&mut draft, kernel, args)
                .map_err(|e| Error::from(CallError::Workflow(e)))?;
            refs.push(
                (0..node.kernel.function.info().results.len())
                    .map(|_| results.take())
                    .collect(),
            );
        }
        let admitted = runtime::admit_workflow(runtime::bind_workflow(draft)?)?;
        for t in moves {
            t.commit();
        }
        let completion = runtime::submit_workflow(admitted)?;
        // Even mutation-only workflows must finish before returning.
        completion.resolve(Vec::new())?;
        resolve(&outputs, &refs, &completion, &mut BTreeMap::new())
    }
}
fn pending_tree(
    ty: &SignatureType,
    elements: &BTreeMap<String, Element>,
    workflow: usize,
    node: usize,
    leaf: &mut usize,
) -> WorkflowValue {
    match ty {
        SignatureType::Unit => WorkflowValue::Unit,
        SignatureType::Tuple(types) => WorkflowValue::Tuple(
            types
                .iter()
                .map(|t| pending_tree(t, elements, workflow, node, leaf))
                .collect(),
        ),
        _ => {
            let mut ty = ty.clone();
            if let SignatureType::Tensor {
                element: ElementSummary::Parameter(name),
                ..
            } = &mut ty
            {
                let name = name.clone();
                if let SignatureType::Tensor { element, .. } = &mut ty {
                    *element = ElementSummary::Fixed(elements[&name].name().into());
                }
            }
            let result = WorkflowValue::Pending(Pending {
                workflow,
                node,
                leaf: *leaf,
                ty,
                slice: None,
            });
            *leaf += 1;
            result
        }
    }
}
fn validate(
    w: &Workflow,
    ty: &SignatureType,
    v: &WorkflowValue,
    elements: &BTreeMap<String, Element>,
    external: &mut BTreeSet<usize>,
    pending: &mut BTreeSet<(usize, usize)>,
) -> Result<(), Error> {
    match (ty, v) {
        (SignatureType::Unit, WorkflowValue::Unit) => Ok(()),
        (SignatureType::Tuple(types), WorkflowValue::Tuple(values))
            if types.len() == values.len() =>
        {
            for (t, v) in types.iter().zip(values) {
                validate(w, t, v, elements, external, pending)?;
            }
            Ok(())
        }
        (_, WorkflowValue::External(v)) => {
            let mut tensors = Vec::new();
            collect_tensors(v, &mut tensors);
            for t in &tensors {
                if external.contains(&(Arc::as_ptr(&t.0) as usize)) {
                    return Err(Error::new(
                        "WorkflowError",
                        "external tensor used after move",
                    ));
                }
            }
            let mut moves = Vec::new();
            encode(
                ty,
                v,
                elements,
                &mut runtime::EncodedArgs::new(),
                &mut moves,
            )?;
            for t in moves {
                external.insert(Arc::as_ptr(&t.0) as usize);
            }
            Ok(())
        }
        (_, WorkflowValue::Pending(p) | WorkflowValue::Move(p)) => {
            if p.workflow != w.id || p.node >= w.nodes.len() {
                return Err(Error::new(
                    "WorkflowError",
                    "pending value belongs to another workflow",
                ));
            }
            if pending.contains(&(p.node, p.leaf)) {
                return Err(Error::new("WorkflowError", "pending value used after move"));
            }
            match (ty, &p.ty) {
                (
                    SignatureType::Tensor {
                        access,
                        rank,
                        element,
                    },
                    SignatureType::Tensor {
                        rank: actual_rank,
                        element: actual_element,
                        ..
                    },
                ) => {
                    let expected = match element {
                        ElementSummary::Fixed(n) => n.as_str(),
                        ElementSummary::Parameter(n) => elements[n].name(),
                    };
                    if rank != actual_rank
                        || actual_element != &ElementSummary::Fixed(expected.into())
                    {
                        return Err(Error::new("TypeError", "pending tensor signature mismatch"));
                    }
                    let owned = *access == TensorAccess::Owned;
                    if owned != matches!(v, WorkflowValue::Move(_)) || (owned && p.slice.is_some())
                    {
                        return Err(Error::new(
                            "TypeError",
                            "pending owned arguments require move of a complete tensor",
                        ));
                    }
                    if owned {
                        pending.insert((p.node, p.leaf));
                    }
                    Ok(())
                }
                _ if ty == &p.ty && matches!(v, WorkflowValue::Pending(_)) => Ok(()),
                _ => Err(Error::new("TypeError", "pending signature mismatch")),
            }
        }
        _ => Err(Error::new(
            "TypeError",
            "workflow argument structure mismatch",
        )),
    }
}
fn collect<'a>(v: &'a WorkflowValue, out: &mut Vec<&'a Tensor>) {
    match v {
        WorkflowValue::External(v) => collect_tensors(v, out),
        WorkflowValue::Tuple(v) => {
            for x in v {
                collect(x, out)
            }
        }
        _ => {}
    }
}
fn encode_workflow(
    ty: &SignatureType,
    v: &WorkflowValue,
    elements: &BTreeMap<String, Element>,
    refs: &[Vec<runtime::WorkflowResultRef>],
    out: &mut runtime::EncodedWorkflowArgs,
    moves: &mut Vec<Tensor>,
) -> Result<(), Error> {
    match v {
        WorkflowValue::External(v) => {
            encode(ty, v, elements, &mut runtime::EncodedArgs::new(), moves)?;
            workflow_encode(v, out)?;
        }
        WorkflowValue::Unit => {}
        WorkflowValue::Tuple(v) => {
            let SignatureType::Tuple(types) = ty else {
                unreachable!()
            };
            for (t, v) in types.iter().zip(v) {
                encode_workflow(t, v, elements, refs, out, moves)?;
            }
        }
        WorkflowValue::Pending(p) | WorkflowValue::Move(p) => {
            let result = refs[p.node][p.leaf];
            if matches!(p.ty, SignatureType::Tensor { .. }) {
                if let Some((start, end)) = p.slice {
                    out.push_result_tensor_view(
                        result,
                        vec![runtime::ViewOperation::LeadingSlice { start, end }],
                    )
                } else {
                    out.push_result_tensor(result)
                }
            } else {
                out.push_result_scalar(result)
            }
        }
    }
    Ok(())
}
fn validate_outputs(w: &Workflow, v: &WorkflowValue) -> Result<(), Error> {
    match v {
        WorkflowValue::Unit => Ok(()),
        WorkflowValue::Tuple(v) => {
            for x in v {
                validate_outputs(w, x)?;
            }
            Ok(())
        }
        WorkflowValue::Pending(p)
            if p.workflow == w.id
                && !w.moved_pending.contains(&(p.node, p.leaf))
                && p.slice.is_none() =>
        {
            Ok(())
        }
        _ => Err(Error::new(
            "WorkflowError",
            "outputs must be unconsumed complete results from this workflow",
        )),
    }
}
fn resolve(
    v: &WorkflowValue,
    refs: &[Vec<runtime::WorkflowResultRef>],
    completion: &runtime::WorkflowCompletionAny,
    cache: &mut BTreeMap<(usize, usize), Value>,
) -> Result<WorkflowValue, Error> {
    Ok(match v {
        WorkflowValue::Unit => WorkflowValue::Unit,
        WorkflowValue::Tuple(v) => WorkflowValue::Tuple(
            v.iter()
                .map(|v| resolve(v, refs, completion, cache))
                .collect::<Result<_, _>>()?,
        ),
        WorkflowValue::Pending(p) => {
            let key = (p.node, p.leaf);
            if !cache.contains_key(&key) {
                let results = completion.resolve(vec![refs[p.node][p.leaf]])?;
                cache.insert(key, decode(&p.ty, &mut results.into_values().into_iter())?);
            }
            WorkflowValue::External(cache[&key].clone())
        }
        _ => unreachable!(),
    })
}
