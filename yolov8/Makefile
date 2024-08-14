CC=g++
CFLAGS=-Wall -Wextra -g -fopenmp
LDFLAGS=-lpthread -lhailort
SCR=inference_hailo8.cpp
TARGET=inference_hailo8
OPENCV_CFLAGS=$(shell pkg-config --cflags opencv4)
OPENCV_LDFLAGS=$(shell pkg-config --libs opencv4)

.PHONY: all

all:
	$(CC) $(CFLAGS) $(OPENCV_CFLAGS) $(SCR) $(LDFLAGS) $(OPENCV_LDFLAGS) -o $(TARGET)