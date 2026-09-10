#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static std::runtime_error failure(NSString* description) {
  return std::runtime_error(description ? description.UTF8String : "Metal operation failed");
}

class Buffer {
 public:
  Buffer(id<MTLDevice> device, size_t size) {
    value_ = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!value_) throw std::bad_alloc();
  }
  ~Buffer() { close(); }
  void close() { if (value_) { [value_ release]; value_ = nil; } }
  id<MTLBuffer> value() const {
    if (!value_) throw std::runtime_error("Metal buffer is closed");
    return value_;
  }
  size_t size() const { return value().length; }
  size_t allocated_bytes() const { return value().allocatedSize; }
  void write(py::bytes content, size_t offset) {
    char* source;
    Py_ssize_t size;
    if (PyBytes_AsStringAndSize(content.ptr(), &source, &size) != 0) throw py::error_already_set();
    bounds(offset, size);
    std::memcpy(static_cast<char*>(value().contents) + offset, source, size);
  }
  py::bytes read(size_t offset, size_t size) const {
    bounds(offset, size);
    return py::bytes(static_cast<const char*>(value().contents) + offset, size);
  }
 private:
  void bounds(size_t offset, size_t size) const {
    if (offset > this->size() || size > this->size() - offset)
      throw std::out_of_range("Metal buffer access exceeds its range");
  }
  id<MTLBuffer> value_ = nil;
};

class Completion {
 public:
  explicit Completion(id<MTLCommandBuffer> buffer) : buffer_([buffer retain]) {}
  ~Completion() { [buffer_ release]; }
  bool ready() const {
    return buffer_.status == MTLCommandBufferStatusCompleted ||
           buffer_.status == MTLCommandBufferStatusError;
  }
  void wait() const {
    [buffer_ waitUntilCompleted];
    if (buffer_.status == MTLCommandBufferStatusError)
      throw failure(buffer_.error.localizedDescription);
    if (buffer_.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error("Metal command buffer did not complete");
  }
 private:
  id<MTLCommandBuffer> buffer_;
};

class Kernel {
 public:
  Kernel(id<MTLDevice> device, const std::string& source, const std::string& name) {
    @autoreleasepool {
      NSString* text = [[NSString alloc] initWithBytes:source.data() length:source.size()
                                             encoding:NSUTF8StringEncoding];
      NSString* symbol = [[NSString alloc] initWithBytes:name.data() length:name.size()
                                               encoding:NSUTF8StringEncoding];
      MTLCompileOptions* options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      id<MTLLibrary> library = [device newLibraryWithSource:text options:options error:&error];
      [text release]; [options release];
      if (!library) { [symbol release]; throw failure(error.localizedDescription); }
      id<MTLFunction> function = [library newFunctionWithName:symbol];
      [symbol release]; [library release];
      if (!function) throw std::runtime_error("compiled Metal entry point is missing");
      pipeline_ = [device newComputePipelineStateWithFunction:function error:&error];
      [function release];
      if (!pipeline_) throw failure(error.localizedDescription);
    }
  }
  ~Kernel() { [pipeline_ release]; }
  id<MTLComputePipelineState> pipeline() const { return pipeline_; }
  size_t subgroup_width() const { return pipeline_.threadExecutionWidth; }
  size_t max_threads() const { return pipeline_.maxTotalThreadsPerThreadgroup; }
 private:
  id<MTLComputePipelineState> pipeline_ = nil;
};

// Concurrent regions use their own encoder. Ordinary work keeps Metal's serial
// encoder behavior; tracked buffers establish dependencies between encoders.
class Encoding {
 public:
  explicit Encoding(id<MTLCommandBuffer> buffer) : buffer_(buffer) {}
  ~Encoding() { finish(); }
  bool independent() const { return independent_; }
  void select(bool independent) {
    if (independent_ != independent) { finish(); independent_ = independent; }
  }
  id<MTLComputeCommandEncoder> current() {
    if (!encoder_) {
      MTLDispatchType type = independent_ ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
      encoder_ = [[buffer_ computeCommandEncoderWithDispatchType:type] retain];
      if (!encoder_) throw std::runtime_error("Metal compute-encoder creation failed");
    }
    return encoder_;
  }
 private:
  void finish() {
    if (encoder_) { [encoder_ endEncoding]; [encoder_ release]; encoder_ = nil; }
  }
  id<MTLCommandBuffer> buffer_;  // Borrowed from Device until record() closes this encoding.
  id<MTLComputeCommandEncoder> encoder_ = nil;
  bool independent_ = false;
};

class EncodingScope {
 public:
  EncodingScope(Encoding& encoding, bool independent)
      : encoding_(encoding), prior_(encoding.independent()) { encoding_.select(independent); }
  ~EncodingScope() { encoding_.select(prior_); }
 private:
  Encoding& encoding_;
  bool prior_;
};

class Dispatch {
 public:
  virtual ~Dispatch() = default;
  virtual void encode(Encoding& encoding) const = 0;
  virtual void encode_bound(Encoding& encoding,
      const std::vector<id<MTLBuffer>>& buffers, const std::vector<size_t>& offsets,
      const std::vector<int>& slots, const std::vector<size_t>& relative) const {
    if (!slots.empty()) throw std::invalid_argument("nested binding has unexpected operands");
    encode(encoding);
  }
};

class Command : public Dispatch {
 public:
  Command(Kernel& kernel, const std::vector<Buffer*>& buffers,
          const std::vector<size_t>& offsets, const std::vector<size_t>& groups,
          const std::vector<size_t>& threads) : pipeline_(kernel.pipeline()), offsets_(offsets) {
    if (buffers.size() != offsets.size() || groups.size() != 3 || threads.size() != 3)
      throw std::invalid_argument("invalid Metal command geometry");
    if (std::any_of(groups.begin(), groups.end(), [](size_t n) { return n == 0; }) ||
        std::any_of(threads.begin(), threads.end(), [](size_t n) { return n == 0; }) ||
        threads[0] * threads[1] * threads[2] > kernel.max_threads())
      throw std::invalid_argument("Metal command exceeds pipeline thread limits");
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (offsets[i] >= buffers[i]->size()) throw std::out_of_range("invalid buffer binding");
      if (buffers[i]->value().device.registryID != pipeline_.device.registryID)
        throw std::invalid_argument("Metal command mixes physical devices");
      buffers_.push_back(buffers[i]->value());
    }
    groups_ = MTLSizeMake(groups[0], groups[1], groups[2]);
    threads_ = MTLSizeMake(threads[0], threads[1], threads[2]);
  }
  void encode(Encoding& encoding) const override {
    auto encoder = encoding.current();
    [encoder setComputePipelineState:pipeline_];
    for (size_t i = 0; i < buffers_.size(); ++i)
      [encoder setBuffer:buffers_[i] offset:offsets_[i] atIndex:i];
    [encoder dispatchThreadgroups:groups_ threadsPerThreadgroup:threads_];
  }
  void encode_bound(Encoding& encoding,
      const std::vector<id<MTLBuffer>>& buffers, const std::vector<size_t>& offsets,
      const std::vector<int>& slots, const std::vector<size_t>& relative) const override {
    auto encoder = encoding.current();
    [encoder setComputePipelineState:pipeline_];
    for (size_t i = 0; i < buffers_.size(); ++i) {
      int slot = slots[i];
      id<MTLBuffer> buffer = slot < 0 ? buffers_[i] : buffers[slot];
      size_t offset = slot < 0 ? offsets_[i] : offsets[slot] + relative[i];
      [encoder setBuffer:buffer offset:offset atIndex:i];
    }
    [encoder dispatchThreadgroups:groups_ threadsPerThreadgroup:threads_];
  }
 private:
  // Borrowed native handles. Prepared/Ticket owns buffer leases and executables
  // before any command reaches the encoder, through proven completion.
  id<MTLComputePipelineState> pipeline_;
  std::vector<id<MTLBuffer>> buffers_;
  std::vector<size_t> offsets_;
  MTLSize groups_, threads_;
};

class Sequence;
class Invocation : public Dispatch {
 public:
  Invocation(std::shared_ptr<Sequence> sequence, const std::vector<Buffer*>& buffers,
             std::vector<size_t> offsets) : sequence_(std::move(sequence)), offsets_(std::move(offsets)) {
    if (buffers.size() != offsets_.size()) throw std::invalid_argument("invalid invocation bindings");
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (offsets_[i] >= buffers[i]->size()) throw std::out_of_range("invalid invocation offset");
      buffers_.push_back(buffers[i]->value());
    }
  }
  void encode(Encoding& encoding) const override;
  void encode_bound(Encoding& encoding,
      const std::vector<id<MTLBuffer>>& buffers, const std::vector<size_t>& offsets,
      const std::vector<int>& slots, const std::vector<size_t>& relative) const override;
 private:
  std::shared_ptr<Sequence> sequence_;
  std::vector<id<MTLBuffer>> buffers_;
  std::vector<size_t> offsets_;
};

class Sequence : public std::enable_shared_from_this<Sequence> {
 public:
  Sequence(std::vector<std::shared_ptr<Dispatch>> commands,
           std::vector<std::vector<int>> slots, std::vector<std::vector<size_t>> relative,
           bool independent)
      : commands_(std::move(commands)), slots_(std::move(slots)), relative_(std::move(relative)),
        independent_(independent) {
    if (commands_.size() != slots_.size() || commands_.size() != relative_.size())
      throw std::invalid_argument("invalid sequence bindings");
  }
  std::shared_ptr<Invocation> bind(const std::vector<Buffer*>& buffers,
                                  std::vector<size_t> offsets) {
    for (const auto& slots : slots_)
      for (int slot : slots)
        if (slot >= 0 && static_cast<size_t>(slot) >= buffers.size())
          throw std::out_of_range("missing invocation operand");
    return std::make_shared<Invocation>(shared_from_this(), buffers, std::move(offsets));
  }
  void encode(Encoding& encoding,
              const std::vector<id<MTLBuffer>>& buffers, const std::vector<size_t>& offsets) const {
    EncodingScope region(encoding, independent_);
    for (size_t i = 0; i < commands_.size(); ++i)
      commands_[i]->encode_bound(encoding, buffers, offsets, slots_[i], relative_[i]);
  }
 private:
  std::vector<std::shared_ptr<Dispatch>> commands_;
  std::vector<std::vector<int>> slots_;
  std::vector<std::vector<size_t>> relative_;
  bool independent_;
};

void Invocation::encode(Encoding& encoding) const {
  sequence_->encode(encoding, buffers_, offsets_);
}

void Invocation::encode_bound(Encoding& encoding,
    const std::vector<id<MTLBuffer>>& buffers, const std::vector<size_t>& offsets,
    const std::vector<int>& slots, const std::vector<size_t>& relative) const {
  if (slots.empty()) { encode(encoding); return; }
  auto bound_buffers = buffers_;
  auto bound_offsets = offsets_;
  for (size_t i = 0; i < slots.size(); ++i) {
    if (slots[i] >= 0) {
      bound_buffers[i] = buffers[slots[i]];
      bound_offsets[i] = offsets[slots[i]] + relative[i];
    }
  }
  sequence_->encode(encoding, bound_buffers, bound_offsets);
}

class Device;
class Interval {
 public:
  Interval(Device& device, id<MTLCommandBuffer> buffer) : device_(device), buffer_([buffer retain]) {}
  ~Interval() { [buffer_ release]; }
  void finish();
  double seconds() const {
    if (!finished_ || buffer_.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error("Metal interval requires finished, completed work");
    double start = buffer_.GPUStartTime, end = buffer_.GPUEndTime;
    if (start <= 0 || end < start) throw std::runtime_error("Metal GPU timestamps unavailable");
    return end - start;
  }
 private:
  Device& device_;
  id<MTLCommandBuffer> buffer_;
  bool finished_ = false;
};

class Device {
 public:
  explicit Device(uint64_t registry) {
    @autoreleasepool {
      if (registry == 0) device_ = MTLCreateSystemDefaultDevice();
      else {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        for (id<MTLDevice> candidate in devices)
          if (candidate.registryID == registry) { device_ = [candidate retain]; break; }
        [devices release];
      }
      if (!device_) throw std::runtime_error("requested Metal device is unavailable");
      queue_ = [device_ newCommandQueue];
      if (!queue_) { [device_ release]; device_ = nil; throw std::runtime_error("Metal queue creation failed"); }
    }
  }
  ~Device() {
    // Destruction is not a completion guarantee; the platform owner drains first.
    encoding_.reset();
    [buffer_ release]; [queue_ release]; [device_ release];
  }
  uint64_t registry_id() const { return device_.registryID; }
  std::shared_ptr<Buffer> allocate(size_t size) {
    if (!size || size > device_.maxBufferLength)
      throw std::invalid_argument("allocation exceeds Metal buffer limits");
    return std::make_shared<Buffer>(device_, size);
  }
  std::shared_ptr<Kernel> compile(const std::string& source, const std::string& name) {
    return std::make_shared<Kernel>(device_, source, name);
  }
  id<MTLCommandBuffer> current() {
    if (!buffer_) {
      // Explicit leases are the single owner of resources consumed by a command.
      buffer_ = [[queue_ commandBufferWithUnretainedReferences] retain];
      if (!buffer_) throw std::runtime_error("Metal command-buffer creation failed");
    }
    return buffer_;
  }
  void dispatch(const std::vector<std::shared_ptr<Dispatch>>& commands) {
    @autoreleasepool {
      if (commands.empty()) return;
      if (!encoding_) encoding_ = std::make_unique<Encoding>(current());
      for (const auto& command : commands) command->encode(*encoding_);
    }
  }
  std::shared_ptr<Completion> record() {
    @autoreleasepool {
      encoding_.reset();
      auto completion = std::make_shared<Completion>(current());
      [buffer_ commit]; [buffer_ release]; buffer_ = nil;
      pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                   [](auto& item) { return item.expired(); }), pending_.end());
      pending_.push_back(completion);
      return completion;
    }
  }
  void drain() {
    if (buffer_) record()->wait();
    for (auto& item : pending_) if (auto completion = item.lock()) completion->wait();
    pending_.clear();
  }
  std::shared_ptr<Interval> interval() {
    drain();
    return std::make_shared<Interval>(*this, current());
  }
 private:
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLCommandBuffer> buffer_ = nil;
  std::unique_ptr<Encoding> encoding_;
  std::vector<std::weak_ptr<Completion>> pending_;
};

void Interval::finish() {
  if (device_.current() != buffer_) throw std::runtime_error("Metal profiling region changed command buffers");
  finished_ = true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<Buffer, std::shared_ptr<Buffer>>(module, "Buffer")
      .def("close", &Buffer::close).def("write", &Buffer::write).def("read", &Buffer::read)
      .def_property_readonly("size", &Buffer::size)
      .def_property_readonly("allocated_bytes", &Buffer::allocated_bytes);
  py::class_<Kernel, std::shared_ptr<Kernel>>(module, "Kernel")
      .def_property_readonly("subgroup_width", &Kernel::subgroup_width)
      .def_property_readonly("max_threads", &Kernel::max_threads);
  py::class_<Dispatch, std::shared_ptr<Dispatch>>(module, "Dispatch");
  py::class_<Invocation, Dispatch, std::shared_ptr<Invocation>>(module, "Invocation");
  py::class_<Sequence, std::shared_ptr<Sequence>>(module, "Sequence")
      .def(py::init<std::vector<std::shared_ptr<Dispatch>>, std::vector<std::vector<int>>,
                   std::vector<std::vector<size_t>>, bool>())
      .def("bind", &Sequence::bind);
  py::class_<Command, Dispatch, std::shared_ptr<Command>>(module, "Command").def(py::init<Kernel&, const std::vector<Buffer*>&,
      const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&>());
  py::class_<Completion, std::shared_ptr<Completion>>(module, "Completion")
      .def("ready", &Completion::ready)
      .def("wait", &Completion::wait, py::call_guard<py::gil_scoped_release>());
  py::class_<Interval, std::shared_ptr<Interval>>(module, "Interval")
      .def("finish", &Interval::finish).def("seconds", &Interval::seconds);
  py::class_<Device>(module, "Device").def(py::init<uint64_t>())
      .def("allocate", &Device::allocate).def("compile", &Device::compile)
      .def("dispatch", &Device::dispatch).def("record", &Device::record)
      .def("drain", &Device::drain, py::call_guard<py::gil_scoped_release>())
      .def("interval", &Device::interval)
      .def_property_readonly("registry_id", &Device::registry_id);
}
