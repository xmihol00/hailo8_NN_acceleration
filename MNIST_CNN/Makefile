CC=g++
CFLAGS=-Wall -Wextra -g
LDFLAGS=-lpthread -lhailort
SCR=inference.cpp
TARGET=inference

.PHONY: all

all:
	$(CC) $(CFLAGS) $(SCR) $(LDFLAGS) -o $(TARGET)