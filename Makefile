# Tabu-Novelty Search — Multi-Platform Makefile
# Usage:
#   make cpu    # builds tns_cpu (any CPU)
#   make arm    # builds tns_arm (ARMv8-A+NEON)
#   make gpu    # builds tns_gpu (CUDA SM_70+)
#   make all    # builds everything available
#   make clean  # removes binaries

CC      = gcc
NVCC    = nvcc
CFLAGS  = -O3 -std=c99 -Wall -Wextra -ffast-math
ARMFLAGS= -march=armv8-a+fp+simd
GPUARCH = -arch=sm_70

SRCDIR  = src
BINDIR  = bin

TARGETS =

# Detect available compilers
HAS_GCC  := $(shell command -v $(CC) 2>/dev/null)
HAS_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

.PHONY: all cpu arm gpu clean dirs

dirs:
	@mkdir -p $(BINDIR)

all: dirs cpu arm gpu

cpu: dirs $(BINDIR)/tns_cpu

arm: dirs $(BINDIR)/tns_arm

gpu: dirs $(BINDIR)/tns_gpu

$(BINDIR)/tns_cpu: $(SRCDIR)/tns_cpu_pure.c
ifeq ($(HAS_GCC),)
	@echo "[SKIP] gcc not found. Install gcc to build CPU target."
else
	$(CC) $(CFLAGS) -march=native -o $@ $< -lm
	@echo "[OK]  CPU target: $@"
endif

$(BINDIR)/tns_arm: $(SRCDIR)/tns_arm_neon.c
ifeq ($(HAS_GCC),)
	@echo "[SKIP] gcc not found. Install gcc to build ARM target."
else
	$(CC) $(CFLAGS) $(ARMFLAGS) -o $@ $< -lm
	@echo "[OK]  ARM target: $@"
endif

$(BINDIR)/tns_gpu: $(SRCDIR)/tns_gpu.cu
ifeq ($(HAS_NVCC),)
	@echo "[SKIP] nvcc not found. Install CUDA toolkit to build GPU target."
else
	$(NVCC) -O3 $(GPUARCH) -o $@ $<
	@echo "[OK]  GPU target: $@"
endif

clean:
	rm -rf $(BINDIR)
