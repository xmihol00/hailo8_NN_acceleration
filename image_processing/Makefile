CC=g++
CFLAGS=-Wall -Wextra -g -fopenmp $(shell pkg-config --cflags --libs opencv4)
LDFLAGS=$(shell pkg-config --cflags --libs opencv4)
SCR_HELLO=openCV_hello_world.cpp
TARGET_HELLO=hello
SCR_DRAW=drawing.cpp
TARGET_DRAW=draw

.PHONY: all hello draw

all: hello draw

hello:
	$(CC) $(CFLAGS) $(SCR_HELLO) $(LDFLAGS) -o $(TARGET_HELLO)

draw:
	$(CC) $(CFLAGS) $(SCR_DRAW) $(LDFLAGS) -o $(TARGET_DRAW)

