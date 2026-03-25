# C shim Makefile
#
# Usage:
#   ORT_HOME=/path/to/onnxruntime make
#
# Output: libort_shim.so (shared) and libort_shim.a (static)

CC      ?= gcc
CFLAGS  ?= -O2 -fPIC -Wall -Wextra
ORT_HOME ?= /usr/local/onnxruntime

INCLUDE = -I$(ORT_HOME)/include
LIBDIR  = -L$(ORT_HOME)/lib

SRC     = ort_shim.c
OBJ     = ort_shim.o
SO      = libort_shim.so
STATIC  = libort_shim.a

.PHONY: all clean

all: $(SO) $(STATIC)

$(OBJ): $(SRC) ort_shim.h
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(SO): $(OBJ)
	$(CC) -shared -o $@ $< $(LIBDIR) -lonnxruntime

$(STATIC): $(OBJ)
	ar rcs $@ $<

clean:
	rm -f $(OBJ) $(SO) $(STATIC)
