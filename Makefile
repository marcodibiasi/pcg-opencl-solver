TARGET = main
SRCS = main.c csr_matrix.c conjugate_gradient.c cbm.h
OBJS = $(SRCS:.c=.o)
CC = gcc

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    OPENCL_FLAGS = -framework OpenCL
else ifeq ($(UNAME_S),Linux)
    OPENCL_FLAGS = -lOpenCL
else ifeq ($(OS),Windows_NT)
    OPENCL_FLAGS = -lOpenCL
    CFLAGS += -I/mingw64/include
    CFLAGS += -L/mingw64/lib
    CFLAGS += -Wno-deprecated-declarations
endif

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(OPENCL_FLAGS)