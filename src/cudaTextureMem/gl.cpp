#include <iostream>
#include <stdlib.h>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define WIDTH 512
#define HEIGHT 512

extern "C" void kernelBindPbo(GLuint pixelBufferObj);
extern "C" void kernelUpdate(int width, int height);
extern "C" void kernelExit(GLuint pixelBufferObj);

GLuint pbo;

int main() {
	if (!glfwInit()) {
		fprintf(stderr, "Failed to init glfw\n");
		return -1;
	}

	GLFWwindow* window;
	window = glfwCreateWindow(WIDTH, HEIGHT, "gl-cuda-test", NULL, NULL);
	if (!window) {
		fprintf(stderr, "Failed to init glfw\n");
		return -1;;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to init glfw\n");
		return -1;
	}

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLubyte) * WIDTH * HEIGHT, NULL, GL_DYNAMIC_DRAW);

	kernelBindPbo(pbo);
	//kernelUpdate(WIDTH, HEIGHT);
	while (!glfwWindowShouldClose(window)) {
		kernelUpdate(WIDTH, HEIGHT);
		glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	kernelExit(pbo);
	glDeleteBuffers(1, &pbo);

	return 0;
}