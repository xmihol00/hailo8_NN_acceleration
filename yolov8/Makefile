CC=g++
CFLAGS=-Wall -Wextra -g -fopenmp --std=c++17
LDFLAGS=-lpthread -lhailort
SCR_INFERENCE=inference_hailo8.cpp
SRC_EVALUATION=evaluation_hailo8.cpp
TARGET_INFERENCE=inference_hailo8
TARGET_EVALUATION=evaluation_hailo8
OPENCV_CFLAGS=$(shell pkg-config --cflags opencv4)
OPENCV_LDFLAGS=$(shell pkg-config --libs opencv4)

.PHONY: all inference evaluation

all: inference evaluation

inference: $(SCR_INFERENCE)
	$(CC) $(CFLAGS) $(OPENCV_CFLAGS) $(SCR_INFERENCE) $(LDFLAGS) $(OPENCV_LDFLAGS) -o $(TARGET_INFERENCE)

evaluation: $(SRC_EVALUATION)
	$(CC) $(CFLAGS) $(OPENCV_CFLAGS) $(SRC_EVALUATION) $(LDFLAGS) $(OPENCV_LDFLAGS) -o $(TARGET_EVALUATION)