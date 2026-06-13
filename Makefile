# ============================================================
# Configuration
# ============================================================

CXX        := g++
CXXFLAGS   := -std=c++17 -O3 -Wall

CUDA_HOME  := /scratch/yrfenach/.fromager/cellars/system
NVCC       := $(CUDA_HOME)/bin/nvcc
ARCH       := sm_70
NVCCFLAGS  := -arch=$(ARCH) \
              -std=c++17 \
              -O3 \
              --expt-relaxed-constexpr \
              --extended-lambda \
              -Xcompiler -Wall

INCLUDES   := -I$(CUDA_HOME)/include -Iinclude
LDFLAGS    := -L$(CUDA_HOME)/lib64 -lcudart -lcublas

CXXFLAGS   += $(INCLUDES)
NVCCFLAGS  += $(INCLUDES)

# ============================================================
# Build rules
# ============================================================
BUILD_DIR  := build
BIN        := $(BUILD_DIR)/cuBench
DBG        := $(BUILD_DIR)/cuDebug

CUDA_SRCS  := $(wildcard src/cuda/*.cu)
CPP_SRCS   := src/bench.cpp
DBG_SRCS   := src/main.cpp

CUDA_OBJS  := $(CUDA_SRCS:src/%.cu=$(BUILD_DIR)/%.o)
CPP_OBJS   := $(CPP_SRCS:src/%.cpp=$(BUILD_DIR)/%.o)
DBG_OBJS   := $(DBG_SRCS:src/%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all debug clean

all: $(BIN)

debug: NVCCFLAGS += -DDEBUG
debug: CXXFLAGS  += -DDEBUG
debug: $(DBG)

$(BIN): $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) $(LDFLAGS) $^ -o $@

$(DBG): $(CUDA_OBJS) $(DBG_OBJS)
	$(NVCC) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/cuda/%.o: src/cuda/%.cu | $(BUILD_DIR)/cuda
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/cuda:
	mkdir -p $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
