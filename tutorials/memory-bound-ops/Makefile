
CUDA_INSTALL_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
TARGET := transpose
CONFIG_NAME ?= SM90_H100
CONFIG_ARCH_SM90_H100 := 90
CONFIG_ARCH_SM90_H200 := 90
CONFIG_ARCH_SM100_B200 := 100a
CONFIG_ARCH_SM120_RTX5090 := 120a
CUDA_ARCH ?= $(CONFIG_ARCH_$(CONFIG_NAME))
ifeq ($(strip $(CUDA_ARCH)),)
$(error Unknown CONFIG_NAME '$(CONFIG_NAME)'; set CUDA_ARCH explicitly)
endif
NVCC_FLAGS ?= -O3 -std=c++17 -lineinfo
GENCODE_FLAGS := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)
.PHONY: all clean
all: $(TARGET)
$(TARGET): transpose.cu
	$(NVCC) $(NVCC_FLAGS) -cudart shared $(GENCODE_FLAGS) $< -o $@
clean:
	$(RM) $(TARGET)
