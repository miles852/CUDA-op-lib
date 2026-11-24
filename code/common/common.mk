# 各算子目录共用的构建规则。算子目录的 Makefile 只需要写：
#   BINS := gemv          # 本目录要编译的可执行文件（同名 .cu）
#   LIBS := -lcublas      # 额外链接的库
#   include ../common/common.mk
#
#   make              编译
#   make test         正确性检查（与 CPU double 参考对比）
#   make bench        性能测试（warmup 20、计时 100、取中位数）
#   make bench-torch  PyTorch 对照组（本目录的 bench_torch.py）
#   make clean        删除可执行文件
# 面向RTX 4090 (sm_89)

COMMON_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
NVCC    ?= nvcc
ARCH    ?= sm_89
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas -v
PYTHON  ?= python3

.PHONY: all test bench bench-torch clean
.DEFAULT_GOAL := all

all: $(BINS)

%: %.cu $(COMMON_DIR)common.cuh
	$(NVCC) $(NVFLAGS) $< -o $@ $(LIBS)

test: $(BINS)
	@for b in $(BINS); do echo "== $$b check"; ./$$b check < /dev/null || exit 1; done

bench: $(BINS)
	@for b in $(BINS); do echo "== $$b bench"; ./$$b bench < /dev/null || exit 1; done

bench-torch:
	@if [ -f bench_torch.py ]; then $(PYTHON) bench_torch.py < /dev/null; fi

clean:
	rm -f $(BINS) *.ncu-rep
