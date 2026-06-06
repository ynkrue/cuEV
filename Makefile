CUDA_HOME  := /scratch/yrfenach/.fromager/cellars/system
NVCC       := $(CUDA_HOME)/bin/nvcc
CXX        := g++

# H200 / Hopper → sm_90a  (use sm_90 if you don't need PTX-level Hopper exts)
ARCH       := sm_90a
GENCODE    := -gencode arch=compute_90a,code=sm_90a

INCLUDES   := -I$(CUDA_HOME)/include -Iinclude
LDFLAGS    := -L$(CUDA_HOME)/lib64 -lcublas -lcudart

NVCCFLAGS  := $(GENCODE) $(INCLUDES) \
              -std=c++17 \
              -O3 \
              --expt-relaxed-constexpr \
              --extended-lambda \
              -Xcompiler -Wall

CXXFLAGS   := -std=c++17 -O3 -Wall $(INCLUDES)

BUILD_DIR  := build
BIN        := $(BUILD_DIR)/cugemv

CUDA_SRCS  := src/cuda/kernels.cu
CPP_SRCS   := src/main.cpp

CUDA_OBJS  := $(CUDA_SRCS:src/%.cu=$(BUILD_DIR)/%.o)
CPP_OBJS   := $(CPP_SRCS:src/%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean

all: $(BIN)

$(BIN): $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) $(GENCODE) $(LDFLAGS) $^ -o $@

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
