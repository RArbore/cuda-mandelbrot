#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <GL/gl.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#define WIDTH 1920
#define HEIGHT 1080
#define PIXELS WIDTH * HEIGHT
#define BYTES_PER_PIXEL 4
#define PIXELS_SIZE PIXELS * BYTES_PER_PIXEL
#define PIXEL_DIV 1024
#define MAX_ITERS 200
#define MOVE_SPEED 10
#define KEY_ESC 27
#define UP 119
#define DOWN 115
#define LEFT 97
#define RIGHT 100

double *x, *y, *zoom;
uint8_t *image;

GLuint tex;
cudaGraphicsResource_t cuda_resource;

__global__
void update_image(uint8_t *image, double *x, double *y, double *zoom) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < PIXELS) {
        int sx = i % WIDTH;
        int sy = i / WIDTH;
        double mx = *x + (sx - WIDTH / 2) / *zoom;
        double my = *y + (sy - HEIGHT / 2) / *zoom;
        double zx = mx;
        double zy = my;
        double zx2, zy2;
        float iters = 0;
        while (zx * zx + zy * zy <= 4.0) {
            zx2 = (zx * zx) - (zy * zy) + mx;
            zy2 = (2 * zx * zy) + my;
            zx = zx2;
            zy = zy2;
            iters++;
            if (iters >= MAX_ITERS) {
                break;
            }
        }
        float H = iters / MAX_ITERS * 210.0 + 15.0;
        float r, g, b;

        float s = 0.5;
        float v = 0.8;
        if (iters >= MAX_ITERS) v = 0.0;
        float C = s*v;
        float X = C*(1-abs(fmod(H/60.0, 2)-1));
        float m = v-C;
        if(H >= 0 && H < 60){
            r = C;
            g = X;
            b = 0;
        }
        else if(H >= 60 && H < 120){
            r = X;
            g = C;
            b = 0;
        }
        else if(H >= 120 && H < 180){
            r = 0;
            g = C;
            b = X;
        }
        else if(H >= 180 && H < 240){
            r = 0;
            g = X;
            b = C;
        }
        else if(H >= 240 && H < 300){
            r = X;
            g = 0;
            b = C;
        }
        else{
            r = C;
            g = 0;
            b = X;
        }
        r = (r+m)*255;
        g = (g+m)*255;
        b = (b+m)*255;

        image[4 * i] = r;
        image[4 * i + 1] = g;
        image[4 * i + 2] = b;
    }
}

__global__
void update_surface(cudaSurfaceObject_t cuda_surface, uint8_t *image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < PIXELS_SIZE) {
        int x = i % (BYTES_PER_PIXEL * WIDTH);
        int y = i / (BYTES_PER_PIXEL * WIDTH);
        surf2Dwrite<uint8_t>(image[i], cuda_surface, x, y);
    }
}

void invokeRenderingKernel(cudaSurfaceObject_t cuda_surface) {
    update_image<<<PIXELS/PIXEL_DIV, PIXEL_DIV>>>(image, x, y, zoom);
    update_surface<<<PIXELS_SIZE/PIXEL_DIV, PIXEL_DIV>>>(cuda_surface, image);
}

void initializeGL () {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    cudaGraphicsGLRegisterImage(&cuda_resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void displayGL() {
    cudaGraphicsMapResources(1, &cuda_resource);
    cudaArray_t cuda_array;
    cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
    cudaResourceDesc cuda_array_resource_desc;
    cuda_array_resource_desc.resType = cudaResourceTypeArray;
    cuda_array_resource_desc.res.array.array = cuda_array;
    cudaSurfaceObject_t cuda_surface;
    cudaCreateSurfaceObject(&cuda_surface, &cuda_array_resource_desc);
    invokeRenderingKernel(cuda_surface);
    cudaDestroySurfaceObject(cuda_surface);
    cudaGraphicsUnmapResources(1, &cuda_resource);

    glBindTexture(GL_TEXTURE_2D, tex);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers();
}

__global__
void set_pos(int mx, int my, double *x, double *y, double *zoom) {
    double set_x = -((double) mx - WIDTH / 2) / *zoom;
    double set_y = ((double) my - HEIGHT / 2) / *zoom;
    *x -= set_x;
    *y -= set_y;
}

__global__
void move(float mx, float my, float mzoom, double *x, double *y, double *zoom) {
    *x += mx / *zoom * MOVE_SPEED;
    *y += my / *zoom * MOVE_SPEED;
    *zoom *= mzoom;
}

void mouse(int button, int state, int mx, int my) {
    if (button == 0 && state == GLUT_DOWN) {
        set_pos<<<1, 1>>>(mx, my, x, y, zoom);
    }
    else if ((button == 3) || (button == 4)) {
       if (state == GLUT_UP) return;
       float mzoom = (button == 3) ? 2.0 : 0.5;
       move<<<1, 1>>>(0.0, 0.0, mzoom, x, y, zoom);
    }
}

void keyboardGL (unsigned char key, int mousePositionX, int mousePositionY) {
    switch (key) {
        case KEY_ESC:
            exit(0);
            break;
        case UP:
            move<<<1, 1>>>(0.0, 1.0, 1.0, x, y, zoom);
            break;
        case DOWN:
            move<<<1, 1>>>(0.0, -1.0, 1.0, x, y, zoom);
            break;
        case LEFT:
            move<<<1, 1>>>(-1.0, 0.0, 1.0, x, y, zoom);
            break;
        case RIGHT:
            move<<<1, 1>>>(1.0, 0.0, 1.0, x, y, zoom);
            break;
        default:
            break;
    }
}

int main (int argc, char *argv[]) {

    srand(time(0));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Mandelbrot Set");
    glutDisplayFunc(displayGL);
    glutIdleFunc(displayGL);
    glutKeyboardFunc(keyboardGL);
    glutMouseFunc(mouse);
    initializeGL();

    int i;

    uint8_t *host_image = (uint8_t *) malloc(PIXELS_SIZE * sizeof(uint8_t));

    double host_x, host_y;
    double host_zoom = HEIGHT / 2;

    cudaMalloc(&image, PIXELS_SIZE * sizeof(uint8_t));
    cudaMalloc(&x, sizeof(double));
    cudaMalloc(&y, sizeof(double));
    cudaMalloc(&zoom, sizeof(double));

    for (i = 0; i < PIXELS_SIZE; i++) {
        host_image[i] = 0;
    }

    cudaMemcpy(image, host_image, PIXELS_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(x, &host_x, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y, &host_y, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(zoom, &host_zoom, sizeof(double), cudaMemcpyHostToDevice);

    glutMainLoop();

    return 0;
}
