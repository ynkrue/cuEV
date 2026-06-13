# ============================================================
# Configuration
# ============================================================

CXX        := g++
CXXFLAGS   := -std=c++17 -O3 -Wall

CUDA_HOME  ?= /scratch/yrfenach/.fromager/cellars/system
NVCC       := $(CUDA_HOME)/bin/nvcc
ARCH       ?= sm_80
NVCCFLAGS  := -arch=$(ARCH) \
              -std=c++17 \
              -O3 \
              --expt-relaxed-constexpr \
              --extended-lambda \
              -Xcompiler -Wall,-fPIC

INCLUDES   := -I$(CUDA_HOME)/include -Iinclude
LDFLAGS    := -L$(CUDA_HOME)/lib64 -lcudart -lcublas

CXXFLAGS   += $(INCLUDES)
NVCCFLAGS  += $(INCLUDES)

# ============================================================
# Sources and objects
# ============================================================
BUILD_DIR   := build
LIB         := $(BUILD_DIR)/libcuev.so
BIN         := $(BUILD_DIR)/cuBench
DBG         := $(BUILD_DIR)/cuDebug

CUDA_SRCS   := $(wildcard src/custom/*.cu)
BENCH_SRCS  := bench/bench.cpp
TEST_SRCS   := test/debug.cpp

CUDA_OBJS   := $(CUDA_SRCS:src/custom/%.cu=$(BUILD_DIR)/custom/%.o)
BENCH_OBJS  := $(BENCH_SRCS:%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJS   := $(TEST_SRCS:%.cpp=$(BUILD_DIR)/%.o)

# ============================================================
# Targets
# ============================================================
.PHONY: all bench debug clean

all: $(LIB)

bench: $(BIN)

debug: NVCCFLAGS += -DDEBUG
debug: CXXFLAGS  += -DDEBUG
debug: $(DBG)

$(LIB): $(CUDA_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $^

$(BIN): $(CUDA_OBJS) $(BENCH_OBJS)
	$(NVCC) $(LDFLAGS) $^ -o $@

$(DBG): $(CUDA_OBJS) $(TEST_OBJS)
	$(NVCC) $(LDFLAGS) $^ -o $@

# ============================================================
# Compile rules
# ============================================================
$(BUILD_DIR)/custom/%.o: src/custom/%.cu | $(BUILD_DIR)/custom
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/bench/%.o: bench/%.cpp | $(BUILD_DIR)/bench
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test/%.o: test/%.cpp | $(BUILD_DIR)/test
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ============================================================
# Directory creation
# ============================================================
$(BUILD_DIR) $(BUILD_DIR)/custom $(BUILD_DIR)/bench $(BUILD_DIR)/test:
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
