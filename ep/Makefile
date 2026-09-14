CUDA_PATH ?= /usr/local/cuda
CXX ?= g++
NVCC      := $(CUDA_PATH)/bin/nvcc

EFA_HOME?=/opt/amazon/efa
ifeq ($(wildcard $(EFA_HOME)),)
  $(warning EFA not detected, building without EFA)
  EFA_CFLAGS :=
  EFA_LDFLAGS :=
else
  $(info EFA detected, building with EFA)
  EFA_CFLAGS := -DEFA -I$(EFA_HOME)/include
  EFA_LDFLAGS := -L$(EFA_HOME)/lib -lefa
endif

# DMA-BUF avoids nvidia_peermem/efa_nv_peermem. Enable with USE_DMABUF=1.
USE_DMABUF ?= 0
ifeq ($(USE_DMABUF),1)
  $(info DMA-BUF GPU memory registration enabled)
  DMABUF_CFLAGS := -DUSE_DMABUF
else
  $(info DMA-BUF GPU memory registration disabled)
  DMABUF_CFLAGS :=
endif

# Enable libfabric/CXI transport skeleton with USE_LIBFABRIC_CXI=1.
USE_LIBFABRIC_CXI ?= 0
ifeq ($(USE_LIBFABRIC_CXI),1)
  $(info libfabric CXI transport enabled)
  CXI_CFLAGS := -DUSE_LIBFABRIC_CXI
  CXI_LDFLAGS := -lfabric
  ifdef LIBFABRIC_HOME
    CXI_CFLAGS += -I$(LIBFABRIC_HOME)/include
    CXI_LDFLAGS += -L$(LIBFABRIC_HOME)/lib
  endif
else
  CXI_CFLAGS :=
  CXI_LDFLAGS :=
endif

ARCH := $(shell uname -m)
GPU_NAME ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
CPU_IS_ARM64 := 0
GPU_IS_HOPPER := 0
ifeq ($(ARCH),aarch64)
  CPU_IS_ARM64 := 1
endif
ifneq (,$(findstring GH200,$(GPU_NAME)))
  GPU_IS_HOPPER := 1
endif
ifeq ($(and $(CPU_IS_ARM64),$(GPU_IS_HOPPER)),0)
  $(warning GH200 not detected, building without GH200)
  GH_CFLAGS :=
else
  $(info GH200 detected, building with GH200)
  GH_CFLAGS := -DUSE_GRACE_HOPPER
endif

# Nanobind configuration
PYTHON             ?= python3
NB_DIR             := $(shell $(PYTHON) -c "import nanobind as _nb, os; print(os.path.dirname(_nb.__file__))")
NB_INCLUDES        := -I$(NB_DIR)/include -I$(NB_DIR)/ext/robin_map/include \
                      -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
# Stable ABI requires Python >= 3.12
PY_STABLE_ABI_OK   := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)")
ifeq ($(PY_STABLE_ABI_OK),1)
NB_ABI_FLAGS       := -DPy_LIMITED_API=0x030C0000 -DNB_STABLE_ABI=1
EXT_SUFFIX         := .abi3.so
else
NB_ABI_FLAGS       :=
EXT_SUFFIX         := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
endif
NB_SRC_DIR         := $(NB_DIR)/src
NB_SRCS            := nb_internals.cpp nb_func.cpp nb_type.cpp nb_enum.cpp \
                      nb_ndarray.cpp nb_static_property.cpp common.cpp error.cpp \
                      trampoline.cpp implicit.cpp
NB_OBJECTS         := $(addprefix nb_,$(NB_SRCS:.cpp=.o))

TORCH_ROOT        := $(shell $(PYTHON) -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent)")
TORCH_INCLUDE     := $(TORCH_ROOT)/include
TORCH_INCLUDE_API := $(TORCH_ROOT)/include/torch/csrc/api/include
TORCH_LIBDIR      := $(TORCH_ROOT)/lib
TORCH_ABI          := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_INCS := -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE_API)

# Python installation path
PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")
INSTALL_DIR          := $(PYTHON_SITE_PACKAGES)/uccl

# GPU arch (auto-detected, override with: make SM=90)
DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM ?= $(DETECTED_SM)

CXXFLAGS  := -O3 -std=c++17 -Wall -pthread -fPIC -fvisibility=hidden -Wno-interference-size -Wno-unused-function -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) -DNB_DOMAIN=uccl_ep
LDFLAGS := -lpthread  -libverbs -lnl-3 -lnl-route-3 -lnuma -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread -fPIC -fvisibility=hidden" -ccbin $(CXX) --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
INCLUDES := -Iinclude -I$(CUDA_PATH)/include -I/usr/include -I../include 

# Add Intel RDMA NIC support if USE_INTEL_RDMA_NIC=1
ifeq ($(USE_INTEL_RDMA_NIC),1)
  override CXXFLAGS += -DINTEL_RDMA_NIC
  override NVCCFLAGS += -DINTEL_RDMA_NIC
endif

# Enable per-expert batching path in dispatch-LL (default: disabled)
PER_EXPERT_BATCHING ?= 0
ifeq ($(PER_EXPERT_BATCHING),1)
  override CXXFLAGS += -DPER_EXPERT_BATCHING
  override NVCCFLAGS += -DPER_EXPERT_BATCHING
endif

# Lane E: dest-rank coalescing dispatch-LL path (default: disabled).
# Independent of PER_EXPERT_BATCHING; do not set both.
LANE_E_DESTRANK_COALESCE ?= 0
ifeq ($(LANE_E_DESTRANK_COALESCE),1)
  ifeq ($(PER_EXPERT_BATCHING),1)
    $(error LANE_E_DESTRANK_COALESCE and PER_EXPERT_BATCHING are mutually exclusive)
  endif
  override CXXFLAGS += -DLANE_E_DESTRANK_COALESCE
  override NVCCFLAGS += -DLANE_E_DESTRANK_COALESCE
endif

# Send one payload per token and remote rank, with a local-expert bitmask.
# All ranks must use the same setting because this changes the staged header.
LANE_E_DISPATCH_DEDUP ?= 0
ifeq ($(LANE_E_DISPATCH_DEDUP),1)
  ifneq ($(LANE_E_DESTRANK_COALESCE),1)
    $(error LANE_E_DISPATCH_DEDUP requires LANE_E_DESTRANK_COALESCE=1)
  endif
  override NVCCFLAGS += -DLANE_E_DISPATCH_DEDUP
endif

# Consume remote payloads directly from staging after publishing slot indices.
LANE_E_DISPATCH_INDIRECT ?= 0
ifeq ($(LANE_E_DISPATCH_INDIRECT),1)
  ifneq ($(LANE_E_DESTRANK_COALESCE),1)
    $(error LANE_E_DISPATCH_INDIRECT requires LANE_E_DESTRANK_COALESCE=1)
  endif
  override NVCCFLAGS += -DLANE_E_DISPATCH_INDIRECT
endif

# Share one dispatch payload across all GPUs on the destination node.
LANE_E_DISPATCH_NODE ?= 0
ifeq ($(LANE_E_DISPATCH_NODE),1)
  ifneq ($(LANE_E_DESTRANK_COALESCE),1)
    $(error LANE_E_DISPATCH_NODE requires LANE_E_DESTRANK_COALESCE=1)
  endif
  ifneq ($(LANE_E_DISPATCH_INDIRECT),1)
    $(error LANE_E_DISPATCH_NODE requires LANE_E_DISPATCH_INDIRECT=1)
  endif
  override CXXFLAGS += -DLANE_E_DISPATCH_NODE
  override NVCCFLAGS += -DLANE_E_DISPATCH_NODE
endif

# Lane E: dest-rank coalescing combine-LL path (default: disabled).
# Intended to be built together with LANE_E_DESTRANK_COALESCE.
LANE_E_COMBINE_COALESCE ?= 0
ifeq ($(LANE_E_COMBINE_COALESCE),1)
  override CXXFLAGS += -DLANE_E_COMBINE_COALESCE
  override NVCCFLAGS += -DLANE_E_COMBINE_COALESCE
endif

# Pack coalesced BF16 combine messages directly from the supplied expert output.
# This removes the redundant full per-expert send arena. Native zero-copy and
# LogFMT inputs are rejected because they require that arena or its transform.
LANE_E_COMBINE_DIRECT_SEND ?= 0
ifeq ($(LANE_E_COMBINE_DIRECT_SEND),1)
  ifneq ($(LANE_E_COMBINE_COALESCE),1)
    $(error LANE_E_COMBINE_DIRECT_SEND requires LANE_E_COMBINE_COALESCE=1)
  endif
  override CXXFLAGS += -DLANE_E_COMBINE_DIRECT_SEND
  override NVCCFLAGS += -DLANE_E_COMBINE_DIRECT_SEND
endif

ifdef NUM_MAX_NVL_PEERS
  override CXXFLAGS += -DNUM_MAX_NVL_PEERS=$(NUM_MAX_NVL_PEERS)
  override NVCCFLAGS += -DNUM_MAX_NVL_PEERS=$(NUM_MAX_NVL_PEERS)
endif

ifdef MAX_NUM_GPUS
  override CXXFLAGS += -DMAX_NUM_GPUS=$(MAX_NUM_GPUS)
  override NVCCFLAGS += -DMAX_NUM_GPUS=$(MAX_NUM_GPUS)
endif


CXXFLAGS  += $(EFA_CFLAGS) $(GH_CFLAGS) $(DMABUF_CFLAGS) $(CXI_CFLAGS) $(NORMAL_CFLAGS)
NVCCFLAGS += $(EFA_CFLAGS) $(GH_CFLAGS) $(DMABUF_CFLAGS) $(CXI_CFLAGS) $(NORMAL_CFLAGS)
LDFLAGS   += $(EFA_LDFLAGS) $(CXI_LDFLAGS)
INCLUDES  += $(EFA_CFLAGS) $(GH_CFLAGS) $(DMABUF_CFLAGS) $(CXI_CFLAGS) $(NORMAL_CFLAGS)

SRC_CPP := src/proxy.cpp src/rdma.cpp src/common.cpp src/uccl_proxy.cpp src/uccl_bench.cpp src/fifo.cpp src/cxi_transport.cpp
SRC_CU  := src/bench_kernel.cu src/internode_ll.cu src/internode.cu src/layout.cu src/intranode.cu src/ep_runtime.cu
# Non-binding .cc translation units linked into the shared library.
SRC_CC_LIB := src/adaptive_sleeper.cc

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)
OBJ_CC_LIB := $(SRC_CC_LIB:.cc=.o)

SRC_BIND := src/uccl_ep.cc
OBJ_BIND := $(SRC_BIND:.cc=.o)
EP_EXT  := ep$(EXT_SUFFIX)
PYTARGET      := ep$(EXT_SUFFIX)

.PHONY: all py clean

all: $(EP_EXT)

# C++ compilation rule with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(NB_ABI_FLAGS) $(TORCH_INCS) $(INCLUDES) $(NB_INCLUDES) -MMD -MP -c $< -o $@
# CUDA compilation rule with dependency generation
%.o: %.cu
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

# Binding source compilation
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(NB_ABI_FLAGS) $(TORCH_INCS) $(INCLUDES) $(NB_INCLUDES) -MMD -MP -c $< -o $@

# Nanobind core sources (compiled with minimal flags)
nb_%.o: $(NB_SRC_DIR)/%.cpp
	$(CXX) -O3 -std=c++17 -fPIC -fvisibility=hidden $(NB_ABI_FLAGS) $(NB_INCLUDES) -c $< -o $@

$(EP_EXT): $(OBJ_CPP) $(OBJ_CU) $(OBJ_CC_LIB) $(OBJ_BIND) $(NB_OBJECTS)
	$(CXX) $(CXXFLAGS) -shared -fPIC \
	    $(INCLUDES) $(NB_INCLUDES) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
	    -ltorch_python -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    -lcuda -lcudart $(LDFLAGS) -o $@

py: $(PYTARGET) $(EP_EXT)

# Repo uccl package dir (so in-repo runs use the same build)
UCCL_UCCL := $(abspath $(CURDIR)/../uccl/uccl)

install: $(EP_EXT)
	@mkdir -p $(INSTALL_DIR)
	@rm -f $(INSTALL_DIR)/ep.cpython-*.so $(INSTALL_DIR)/ep.abi3.so
	@cp $(EP_EXT) $(INSTALL_DIR)/
	@echo "Installed $(EP_EXT) -> $(INSTALL_DIR)/"
	@if [ -d "$(UCCL_UCCL)" ]; then \
		rm -f "$(UCCL_UCCL)"/ep.cpython-*.so "$(UCCL_UCCL)"/ep.abi3.so; \
		cp $(EP_EXT) "$(UCCL_UCCL)/"; \
		echo "Installed $(EP_EXT) -> $(UCCL_UCCL)/ (for in-repo runs)"; \
	fi

# Clean all generated files
clean:
	rm -f $(OBJ_CPP) $(OBJ_CC) $(OBJ_CU) $(OBJ_CC_LIB) $(OBJ_BIND) $(NB_OBJECTS) \
	      $(EP_EXT) \
	      *.d src/*.d bench/*.d src/*.o **/*.hip **/*_hip.*

# Automatically include dependency files if they exist
DEPS := $(OBJ_CPP:.o=.d) $(OBJ_CC:.o=.d) $(OBJ_CC_LIB:.o=.d) $(OBJ_CU:.o=.d) $(OBJ_BIND:.o=.d)
-include $(DEPS)

# Consume remote expert outputs directly from the coalesced staging buffer.
LANE_E_COMBINE_INDIRECT ?= 0
ifeq ($(LANE_E_COMBINE_INDIRECT),1)
  ifneq ($(LANE_E_COMBINE_COALESCE),1)
    $(error LANE_E_COMBINE_INDIRECT requires LANE_E_COMBINE_COALESCE=1)
  endif
  override NVCCFLAGS += -DLANE_E_COMBINE_INDIRECT
endif
