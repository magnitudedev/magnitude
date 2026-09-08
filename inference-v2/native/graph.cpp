// Graph inspection only. Numerical execution remains in the original MLX primitive.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <mlx/array.h>
#include <mlx/primitives.h>
#include <functional>
#include <unordered_set>
namespace nb = nanobind;
namespace mx = mlx::core;

struct Node {
  mx::array result;
  std::vector<mx::array> inputs() const { return result.inputs(); }
  std::vector<mx::array> outputs() const { return result.outputs(); }
  std::string name() const {
    auto name = std::string(result.primitive().name());
    return name == "Rsqrt" ? "Sqrt" : name;
  }
  nb::tuple arguments() const {
    auto& primitive = result.primitive();
    if (name() == "Sqrt") return nb::make_tuple(std::string(primitive.name()) == "Rsqrt");
    if (name() == "AsType") return nb::make_tuple(result.dtype());
    if (name() == "Broadcast" || name() == "Full") {
      nb::list shape; for (auto extent : result.shape()) shape.append(extent);
      return nb::make_tuple(nb::tuple(shape));
    }
    return nb::tuple();
  }
  std::vector<mx::array> apply(const std::vector<mx::array>& inputs) const {
    if (inputs.size() != result.inputs().size())
      throw std::invalid_argument("retained primitive operand count changed");
    auto outputs = result.outputs();
    std::vector<mx::Shape> shapes;
    std::vector<mx::Dtype> dtypes;
    for (const auto& output : outputs) {
      shapes.push_back(output.shape());
      dtypes.push_back(output.dtype());
    }
    // Reuse the original primitive, including its stream and operation attributes.
    return mx::array::make_arrays(shapes, dtypes, result.primitive_ptr(), inputs);
  }
};

NB_MODULE(_graph, m) {
  m.attr("source") = nb::str(MAGNITUDE_GRAPH_SOURCE);
  m.def("in_tracing", []() { mx::array probe(0); probe.set_tracer(true); return probe.is_tracer(); });
  m.def("identity", [](const mx::array& a) { return a.id(); });
  m.def("is_tracer", [](const mx::array& a) { return a.is_tracer(); });
  nb::class_<Node>(m, "Node")
    .def_prop_ro("inputs", &Node::inputs)
    .def_prop_ro("outputs", &Node::outputs)
    .def_prop_ro("name", &Node::name)
    .def_prop_ro("arguments", &Node::arguments)
    .def_prop_ro("stream", [](const Node& n) { return n.result.primitive().stream(); })
    .def("apply", &Node::apply);
  m.def("inspect", [](const std::vector<mx::array>& outputs,
                       const std::vector<mx::array>& boundaries) {
    std::unordered_set<std::uintptr_t> seen;
    for (const auto& a : boundaries) seen.insert(a.id());
    std::vector<mx::array> leaves;
    std::vector<Node> nodes;
    std::function<void(const mx::array&)> visit = [&](const mx::array& a) {
      if (seen.count(a.id())) return;
      if (!a.has_primitive() || !a.is_tracer()) {
        seen.insert(a.id()); leaves.push_back(a); return;
      }
      for (const auto& input : a.inputs()) visit(input);
      for (const auto& output : a.outputs()) seen.insert(output.id());
      nodes.push_back(Node{a});
    };
    for (const auto& output : outputs) visit(output);
    return nb::make_tuple(nodes, leaves);
  });
}
