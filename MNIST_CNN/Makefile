CC=g++
CFLAGS=-Wall -Wextra -g -fopenmp
LDFLAGS=-lpthread -lhailort
SCR_SYNC=inference.cpp
SCR_ASYNC=inference_async.cpp
TARGET_SYN=inference
TARGET_ASYNC=inference_async

.PHONY: all

all: sync async	

sync:
	$(CC) $(CFLAGS) $(SCR_SYNC) $(LDFLAGS) -o $(TARGET_SYN)

async:
	$(CC) $(CFLAGS) $(SCR_ASYNC) $(LDFLAGS) -o $(TARGET_ASYNC)