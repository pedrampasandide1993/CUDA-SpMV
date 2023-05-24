CC=nvcc
CFLAGS=-arch=sm_61 -lgsl -lgslcblas -Xcompiler -fopenmp

SRCS=main.cu
OBJS=$(subst .cu,.o,$(SRCS))
EXEC=main

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(EXEC) $(OBJS)
