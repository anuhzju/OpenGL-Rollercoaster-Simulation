/*
  CSCI 420 Computer Graphics, Computer Science, USC
  Assignment 2: Roller Coaster
  C/C++ starter code

  Student username: annazhu
*/

#include "openGLHeader.h"
#include "glutHeader.h"
#include "openGLMatrix.h"
#include "imageIO.h"
#include "pipelineProgram.h"
#include "vbo.h"
#include "vao.h"

#include <iostream>
#include <cstring>
#include <memory>
#include <cmath>
#include <vector>

#if defined(WIN32) || defined(_WIN32)
  #ifdef _DEBUG
    #pragma comment(lib, "glew32d.lib")
  #else
    #pragma comment(lib, "glew32.lib")
  #endif
#endif

#if defined(WIN32) || defined(_WIN32)
  char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
  char shaderBasePath[1024] = "../openGLHelper";
#endif

using namespace std;

int mousePos[2]; // x,y screen coordinates of the current mouse position

int leftMouseButton = 0; // 1 if pressed, 0 if not 
int middleMouseButton = 0; // 1 if pressed, 0 if not
int rightMouseButton = 0; // 1 if pressed, 0 if not

typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;
CONTROL_STATE controlState = ROTATE;

// Transformations of the terrain.
float terrainRotate[3] = { 0.0f, 0.0f, 0.0f }; 
// terrainRotate[0] gives the rotation around x-axis (in degrees)
// terrainRotate[1] gives the rotation around y-axis (in degrees)
// terrainRotate[2] gives the rotation around z-axis (in degrees)
float terrainTranslate[3] = { 0.0f, 0.0f, 0.0f };
float terrainScale[3] = { 1.0f, 1.0f, 1.0f };

// camera
float cameraYaw = 0.0f;
float cameraPitch = 0.0f;
float mouseSensitivity = 0.2f;

// Width and height of the OpenGL window, in pixels.
int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "CSCI 420 Homework 1";


// Number of vertices in the single triangle (starter code).
int numVertices;

// CSCI 420 helper classes.
OpenGLMatrix matrix;
PipelineProgram pipelineProgram;
PipelineProgram textureProgram; // texture shader for ground
VBO vboVertices;
VBO vboColors;
VAO vao;

//spline globals
VBO vboSplinePos, vboSplineCol;
VAO vaoSpline;
int splineVertexCount = 0;
float trackS = 0.0f;
float rideSpeed = 0.2f;
int lastSeg = 1;

// Debug spline visualization
PipelineProgram debugProgram;
VAO vaoDebugSpline, vaoDebugControlPoints;
VBO vboDebugSplinePos, vboDebugControlPos;
int debugSplineVertexCount = 0;
int debugControlPointCount = 0;

bool showSplineDebug = true;

// spline control point struct
struct Point
{
    float x, y, z;
};

// spline struct
struct Spline
{
    int numControlPoints;
    Point* points;
} spline;

struct Vec3 
{ 
    float x, y, z; 
};

Vec3  prevUp = { 0,1,0 };

//spline helpers
inline Vec3 toV(const Point& p) { return { p.x, p.y, p.z }; }
inline Vec3 add(Vec3 a, Vec3 b) { return { a.x + b.x,a.y + b.y,a.z + b.z }; }
inline Vec3 sub(Vec3 a, Vec3 b) { return { a.x - b.x,a.y - b.y,a.z - b.z }; }
inline Vec3 mul(Vec3 a, float s) { return { a.x * s,a.y * s,a.z * s }; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return { a.y * b.z - a.z * b.y, 
                                             a.z * b.x - a.x * b.z, 
                                             a.x * b.y - a.y * b.x }; }
inline float len(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 nrm(Vec3 a) { float L = len(a); return (L > 1e-6f) ? mul(a, 1.0f / L) : Vec3{ 0,0,0 }; }

void loadSpline(char* argv)
{
    FILE* fileSpline = fopen(argv, "r");
    if (fileSpline == NULL)
    {
        printf("Cannot open file %s.\n", argv);
        exit(1);
    }

    // read spline control points
    fscanf(fileSpline, "%d\n", &spline.numControlPoints);
    printf("Detected %d control points.\n", spline.numControlPoints);

    spline.points = (Point*)malloc(spline.numControlPoints * sizeof(Point));
    // load control points.
    for (int i = 0; i < spline.numControlPoints; i++)
    {
        if (fscanf(fileSpline, "%f %f %f",
            &spline.points[i].x,
            &spline.points[i].y,
            &spline.points[i].z) != 3)
        {
            printf("Error: incorrect number of control points in file %s.\n", argv);
            exit(1);
        }
    }
}

// convert spline parameter s to segment index seg and local parameter u
void paramToSegU(float s, int& seg, float& u, int N)
{
    float maxSeg = float(N - 3);
    // wrap around loop
    float sWrapped = std::fmod(s, maxSeg);
    if (sWrapped < 0.0f) sWrapped += maxSeg; 
    seg = int(std::floor(sWrapped)) + 1;
    u = sWrapped - float(seg - 1);
    // clamp in case of tiny fp errors
    if (seg >= N - 2) { seg = N - 3; u = 1.0f; }
    if (seg < 1) { seg = 1; u = 0.0f; }
    // ensure u stays in [0, 1)
    if (u >= 1.0f) { u = 0.9999f; }
    if (u < 0.0f) { u = 0.0f; }
}

// parallel-transport-ish frame update.
void buildFrame(const Vec3& T, Vec3& U /*in/out*/, Vec3& B /*out*/)
{
    // pick a safe fallback if previous U was unset or nearly parallel to T
    if (len(U) < 1e-6f || std::abs(dot(U, T)) > 0.99f)
        U = std::abs(T.y) < 0.9f ? Vec3{ 0,1,0 } : Vec3{ 1,0,0 };

    // project previous up onto plane orthogonal to T
    U = sub(U, mul(T, dot(U, T)));
    U = nrm(U);

    // build binormal and re-orthonormalize
    B = nrm(cross(T, U));
    U = nrm(cross(B, T));
}

// Multiply C = A * B, where A is a m x p matrix, and B is a p x n matrix.
// All matrices A, B, C must be pre-allocated (say, using malloc or similar).
// The memory storage for C must *not* overlap in memory with either A or B. 
// That is, you **cannot** do C = A * C, or C = C * B. However, A and B can overlap, and so C = A * A is fine, as long as the memory buffer for A is not overlaping in memory with that of C.
// Very important: All matrices are stored in **column-major** format.
// Example. Suppose 
//      [ 1 8 2 ]
//  A = [ 3 5 7 ]
//      [ 0 2 4 ]
//  Then, the storage in memory is
//   1, 3, 0, 8, 5, 2, 2, 7, 4. 
void MultiplyMatrices(int m, int p, int n, const float* A, const float* B, float* C)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float entry = 0.0;
            for (int k = 0; k < p; k++)
                entry += A[k * m + i] * B[j * p + k];
            C[m * j + i] = entry;
        }
    }
}

// s = 0.5 catmull–rom blending (uniform)
inline Point crEval(const Point& Pm1, const Point& P0, const Point& P1, const Point& P2, float u)
{
    float u2 = u * u, u3 = u2 * u;
    float b0 = -0.5f * u3 + 1.0f * u2 - 0.5f * u;
    float b1 = 1.5f * u3 - 2.5f * u2 + 1.0f;
    float b2 = -1.5f * u3 + 2.0f * u2 + 0.5f * u;
    float b3 = 0.5f * u3 - 0.5f * u2;

    Point p;
    p.x = b0 * Pm1.x + b1 * P0.x + b2 * P1.x + b3 * P2.x;
    p.y = b0 * Pm1.y + b1 * P0.y + b2 * P1.y + b3 * P2.y;
    p.z = b0 * Pm1.z + b1 * P0.z + b2 * P1.z + b3 * P2.z; 

    return p;
}

// evaluate the tangent vector of the spline at parameter u
inline Point crTangent(const Point& Pm1, const Point& P0, const Point& P1, const Point& P2, float u)
{
    float u2 = u * u;
    float db0 = -1.5f * u2 + 2.0f * u - 0.5f;
    float db1 = 4.5f * u2 - 5.0f * u;
    float db2 = -4.5f * u2 + 4.0f * u + 0.5f;
    float db3 = 1.5f * u2 - 1.0f * u;

    Point t;
    t.x = db0 * Pm1.x + db1 * P0.x + db2 * P1.x + db3 * P2.x;
    t.y = db0 * Pm1.y + db1 * P0.y + db2 * P1.y + db3 * P2.y;
    t.z = db0 * Pm1.z + db1 * P0.z + db2 * P1.z + db3 * P2.z;
    return t;
}

int initTexture(const char* imageFilename, GLuint textureHandle)
{
    // read texture image.
    ImageIO img;
    ImageIO::fileFormatType imgFormat;
    ImageIO::errorType err = img.load(imageFilename, &imgFormat);

    if (err != ImageIO::OK)
    {
        printf("Loading texture from %s failed.\n", imageFilename);
        return -1;
    }

    if (img.getWidth() * img.getBytesPerPixel() % 4)
    {
        printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
        return -1;
    }

    int width = img.getWidth();
    int height = img.getHeight();
    unsigned char* pixelsRGBA = new unsigned char[4 * width * height]; 

    memset(pixelsRGBA, 0, 4 * width * height); // set all bytes to 0
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
        {
            const int base = 4 * (h * width + w);
            pixelsRGBA[base + 0] = 0;
            pixelsRGBA[base + 1] = 0;
            pixelsRGBA[base + 2] = 0;
            pixelsRGBA[base + 3] = 255;

            int numChannels = img.getBytesPerPixel();
            if (numChannels == 1)
            {
                unsigned char v = img.getPixel(w, h, 0);
                pixelsRGBA[base + 0] = v;
                pixelsRGBA[base + 1] = v;
                pixelsRGBA[base + 2] = v;
            }
            else
            {
                for (int c = 0; c < numChannels; c++)
                    pixelsRGBA[base + c] = img.getPixel(w, h, c);
            }
        }

    // bind texture
    glBindTexture(GL_TEXTURE_2D, textureHandle);

    // initialize texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);

    // generate mipmaps for this texture
    glGenerateMipmap(GL_TEXTURE_2D);

    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // query support for anisotropic texture filtering
    GLfloat fLargest;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
    printf("Max available anisotropic samples: %f\n", fLargest);

    // set anisotropic texture filtering
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);

    // query for any errors
    GLenum errCode = glGetError();
    if (errCode != 0)
    {
        printf("Texture initialization error. Error code: %d.\n", errCode);
        return -1;
    }

    // de-allocate the pixel array -- it is no longer needed
    delete[] pixelsRGBA;

    return 0;
}


// global params for different render modes
enum class RenderMode { Points = 1, Grid = 2, Tris = 3, Smooth = 4 };
RenderMode rm = RenderMode::Points;
int imageW = 0, imageH = 0;
const unsigned char* pixels = nullptr;
void geometryForMode();
VBO vboLeft, vboRight, vboUp, vboDown;
float uScale = 0.1f;
float uExponent = 1.0f;
int colormapMode = 0;

// ground stuff
VAO vaoGround;
VBO vboGroundPos, vboGroundTex;
GLuint groundTexture = 0;

// skybox
PipelineProgram skyboxProgram;
VAO vaoSkybox;
VBO vboSkyboxPos;
GLuint skyboxTexture = 0;

// support pillars
VAO vaoPillars;
VBO vboPillarPos, vboPillarNormals;
int pillarVertexCount = 0;

// pillar settings
float pillarSpacing = 6.0f;
float pillarWidth = 0.08f;
float pillarGroundY = -1.0f;
float pillarTopOffset = 0.07f;
float pillarSpread = 1.0f;  // distance between ground contact points

// Write a screenshot to the specified filename.
void saveScreenshot(const char * filename)
{
  std::unique_ptr<unsigned char[]> screenshotData = std::make_unique<unsigned char[]>(windowWidth * windowHeight * 3);
  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData.get());

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData.get());

  if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
    cout << "File " << filename << " saved successfully." << endl;
  else cout << "Failed to save file " << filename << '.' << endl;
}

void idleFunc()
{
  // Do some stuff... 
  // For example, here, you can save the screenshots to disk (to make the animation).

  // Notify GLUT that it should call displayFunc.
  glutPostRedisplay();
}

void reshapeFunc(int w, int h)
{
  glViewport(0, 0, w, h);

  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.LoadIdentity();

  const float zNear = 0.1f;
  const float zFar = 10000.0f;
  matrix.Perspective(60.0f, 1.0f * w / h, zNear, zFar);
}

void mouseMotionDragFunc(int x, int y)
{
  // Mouse has moved, and one of the mouse buttons is pressed (dragging).

  // the change in mouse position since the last invocation of this function
  int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };

  switch (controlState)
  {
    // translate the terrain
    case TRANSLATE:
      if (leftMouseButton)
      {
        // control x,y translation via the left mouse button
        // drag right moves right, drag up moves up (screen space)
        terrainTranslate[0] += mousePosDelta[0] * 0.01f;
        terrainTranslate[1] += mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z translation via the middle mouse button
        terrainTranslate[2] -= mousePosDelta[1] * 0.01f;
      }
      break;

    // rotate the terrain
    case ROTATE:
      if (leftMouseButton)
      {
          cameraYaw += mousePosDelta[0] * mouseSensitivity;
          cameraPitch -= mousePosDelta[1] * mouseSensitivity;

          cameraPitch = std::max(-89.0f,
                          std::min(89.0f, cameraPitch));
      }
      break;

    // scale the terrain
    case SCALE:
      if (leftMouseButton)
      {
        // control x,y scaling via the left mouse button
        terrainScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
        terrainScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z scaling via the middle mouse button
        terrainScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      break;
  }

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseMotionFunc(int x, int y)
{
  // Mouse has moved.
  // Store the new mouse position.
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseButtonFunc(int button, int state, int x, int y)
{
  // A mouse button has has been pressed or depressed.

  // Keep track of the mouse button state, in leftMouseButton, middleMouseButton, rightMouseButton variables.
  switch (button)
  {
    case GLUT_LEFT_BUTTON:
      leftMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_MIDDLE_BUTTON:
      middleMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_RIGHT_BUTTON:
      rightMouseButton = (state == GLUT_DOWN);
    break;
  }

  // Keep track of whether CTRL and SHIFT keys are pressed.
  switch (glutGetModifiers())
  {
    case GLUT_ACTIVE_CTRL:
      controlState = TRANSLATE;
    break;

    case GLUT_ACTIVE_SHIFT:
      controlState = SCALE;
    break;

    // If CTRL and SHIFT are not pressed, we are in rotate mode.
    default:
      controlState = ROTATE;
    break;
  }

  // Store the new mouse position.
  mousePos[0] = x;
  mousePos[1] = y;
}

void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 27: // ESC key
      exit(0); // exit the program
    break;

    case ' ':
      cout << "You pressed the spacebar." << endl;
    break;

    case 'x':
      // Take a screenshot.
      saveScreenshot("screenshot.jpg");
    break;

    //switching render modes
    case '1': 
        rm = RenderMode::Points;
        break;
    case '2':
        rm = RenderMode::Grid;
        break;
    case '3':
        rm = RenderMode::Tris;
        break;
    case '4':
        rm = RenderMode::Smooth;
        break;

    //scale&exponent manip for smooth mode
    case '=':
        uScale *= 2.0;
        break;
    case '-':
        uScale /= 2.0;
        break;
    case '9':
        uExponent *= 2.0;
        break;
    case '0':
        uExponent /= 2.0;
        break;

    //cycles through additional colormaps
    case ']':
        if (colormapMode < 4)
            colormapMode++;
        else
            colormapMode = 0;
        break;
    case '[':
        if (colormapMode > 0)
            colormapMode--;
        else
            colormapMode = 4;
        break;
    // debug spline display
    case 'p':
      showSplineDebug = !showSplineDebug;
      break;
  }
}

void buildSplineGeometry(int samplesPerSeg = 500)
{
    const int N = spline.numControlPoints;
    if (N < 4) {
        splineVertexCount = 0;
        return;
    }

    // T-shaped rail params
    const float flangeHalfWidth = 0.05f;   // half width of top flat
    const float flangeThickness = 0.02f;   // thickness of the top flat
    const float stemHalfWidth = 0.02f;   // half width of the vertical stem
    const float stemHeight = 0.06f;   // height of the vertical stem (downwards)

    // number of segments along the spline
    const int numSegs = (N - 3) * samplesPerSeg;
    const int trisPerSeg = (4 * 2) * 2; // 16 tris/segment
    const int vertsPerSeg = trisPerSeg * 3; // 48 verts/segment
    const int totalVerts = numSegs * vertsPerSeg;
    splineVertexCount = totalVerts;

    std::unique_ptr<float[]> positions(new float[totalVerts * 3]);
    std::unique_ptr<float[]> colors(new float[totalVerts * 4]);

    auto evalPoint = [&](int segIdx, float u) -> Point {
        const Point& Qm1 = spline.points[segIdx - 1];
        const Point& Q0  = spline.points[segIdx + 0];
        const Point& Q1  = spline.points[segIdx + 1];
        const Point& Q2  = spline.points[segIdx + 2];
        return crEval(Qm1, Q0, Q1, Q2, u);
    };

    auto evalTangent = [&](int segIdx, float u) -> Vec3 {
        const Point& Qm1 = spline.points[segIdx - 1];
        const Point& Q0  = spline.points[segIdx + 0];
        const Point& Q1  = spline.points[segIdx + 1];
        const Point& Q2  = spline.points[segIdx + 2];
        Point t = crTangent(Qm1, Q0, Q1, Q2, u);
        return nrm({ t.x, t.y, t.z });
    };

    int v = 0; // vertex cursor
    Vec3 U = { 0,1,0 };
    Vec3 B = { 1,0,0 };

    auto emitTri = [&](const Vec3& a, const Vec3& b, const Vec3& c)
    {
        Vec3 n = nrm(cross(sub(b, a), sub(c, a)));
        positions[3 * v + 0] = a.x; positions[3 * v + 1] = a.y; positions[3 * v + 2] = a.z;
        colors[4 * v + 0] = n.x; colors[4 * v + 1] = n.y; colors[4 * v + 2] = n.z; colors[4 * v + 3] = 1.0f; ++v;
        positions[3 * v + 0] = b.x; positions[3 * v + 1] = b.y; positions[3 * v + 2] = b.z;
        colors[4 * v + 0] = n.x; colors[4 * v + 1] = n.y; colors[4 * v + 2] = n.z; colors[4 * v + 3] = 1.0f; ++v;
        positions[3 * v + 0] = c.x; positions[3 * v + 1] = c.y; positions[3 * v + 2] = c.z;
        colors[4 * v + 0] = n.x; colors[4 * v + 1] = n.y; colors[4 * v + 2] = n.z; colors[4 * v + 3] = 1.0f; ++v;
    };

    auto emitQuad = [&](const Vec3& r0a, const Vec3& r1a, const Vec3& r0b, const Vec3& r1b)
    {
        // two triangles: (r0a, r1a, r0b) and (r0b, r1a, r1b)
        emitTri(r0a, r1a, r0b);
        emitTri(r0b, r1a, r1b);
    };

    for (int segIdx = 1; segIdx <= N - 3; ++segIdx)
    {
        for (int s = 0; s < samplesPerSeg; ++s)
        {
            float u0 = float(s) / float(samplesPerSeg);
            float u1 = float(s + 1) / float(samplesPerSeg);

            Point p0 = evalPoint(segIdx, u0);
            Point p1 = evalPoint(segIdx, u1);

            Vec3 P0 = { p0.x, p0.y, p0.z };
            Vec3 P1 = { p1.x, p1.y, p1.z };

            Vec3 T0 = evalTangent(segIdx, u0);
            buildFrame(T0, U, B); // updates U and B
            Vec3 U0 = U, B0 = B;

            Vec3 T1 = evalTangent(segIdx, u1);
            buildFrame(T1, U, B);
            Vec3 U1 = U, B1 = B;

            //flange/stem ring corners
            float fuTop  = +flangeThickness * 0.5f;
            float fuBot  = -flangeThickness * 0.5f;
            float suTop  = fuBot;                 
            float suBot  = fuBot - stemHeight;  

            // flange ring 0 corners (B then U)
            Vec3 f0_bl = add(P0, add(mul(B0, -flangeHalfWidth), mul(U0, fuBot)));
            Vec3 f0_br = add(P0, add(mul(B0,  flangeHalfWidth), mul(U0, fuBot)));
            Vec3 f0_tr = add(P0, add(mul(B0,  flangeHalfWidth), mul(U0, fuTop)));
            Vec3 f0_tl = add(P0, add(mul(B0, -flangeHalfWidth), mul(U0, fuTop)));

            // flange ring 1 corners
            Vec3 f1_bl = add(P1, add(mul(B1, -flangeHalfWidth), mul(U1, fuBot)));
            Vec3 f1_br = add(P1, add(mul(B1,  flangeHalfWidth), mul(U1, fuBot)));
            Vec3 f1_tr = add(P1, add(mul(B1,  flangeHalfWidth), mul(U1, fuTop)));
            Vec3 f1_tl = add(P1, add(mul(B1, -flangeHalfWidth), mul(U1, fuTop)));

            // stem ring 0 corners
            Vec3 s0_bl = add(P0, add(mul(B0, -stemHalfWidth), mul(U0, suBot)));
            Vec3 s0_br = add(P0, add(mul(B0,  stemHalfWidth), mul(U0, suBot)));
            Vec3 s0_tr = add(P0, add(mul(B0,  stemHalfWidth), mul(U0, suTop)));
            Vec3 s0_tl = add(P0, add(mul(B0, -stemHalfWidth), mul(U0, suTop)));

            // stem ring 1 corners
            Vec3 s1_bl = add(P1, add(mul(B1, -stemHalfWidth), mul(U1, suBot)));
            Vec3 s1_br = add(P1, add(mul(B1,  stemHalfWidth), mul(U1, suBot)));
            Vec3 s1_tr = add(P1, add(mul(B1,  stemHalfWidth), mul(U1, suTop)));
            Vec3 s1_tl = add(P1, add(mul(B1, -stemHalfWidth), mul(U1, suTop)));

            // flange faces
            emitQuad(f0_br, f1_br, f0_tr, f1_tr); // +B face
            emitQuad(f0_tl, f1_tl, f0_bl, f1_bl); // -B face (order to keep outward)
            emitQuad(f0_tr, f1_tr, f0_tl, f1_tl); // +U top face
            emitQuad(f0_bl, f1_bl, f0_br, f1_br); // -U bottom face

            // stem faces
            emitQuad(s0_br, s1_br, s0_tr, s1_tr); // +B face
            emitQuad(s0_tl, s1_tl, s0_bl, s1_bl); // -B face
            emitQuad(s0_tr, s1_tr, s0_tl, s1_tl); // +U top face (meets flange bottom)
            emitQuad(s0_bl, s1_bl, s0_br, s1_br); // -U bottom face
        }
    }

    // upload to GPU
    vboSplinePos.Gen(splineVertexCount, 3, positions.get(), GL_STATIC_DRAW);
    vboSplineCol.Gen(splineVertexCount, 4, colors.get(), GL_STATIC_DRAW);

    vaoSpline.Gen();
    vaoSpline.ConnectPipelineProgramAndVBOAndShaderVariable(&pipelineProgram, &vboSplinePos, "position");
    vaoSpline.ConnectPipelineProgramAndVBOAndShaderVariable(&pipelineProgram, &vboSplineCol, "normal");
}

void buildDebugSplineGeometry(int samplesPerSeg = 200)
{
    const int N = spline.numControlPoints;
    if (N < 4)
        return;

    // --- Control points ---
    std::vector<float> controlPositions;
    controlPositions.reserve(N * 3);

    for (int i = 0; i < N; ++i)
    {
        controlPositions.push_back(spline.points[i].x);
        controlPositions.push_back(spline.points[i].y);
        controlPositions.push_back(spline.points[i].z);
    }

    debugControlPointCount = N;

    vboDebugControlPos.Gen(
        debugControlPointCount,
        3,
        controlPositions.data(),
        GL_STATIC_DRAW
    );

    vaoDebugControlPoints.Gen();
    vaoDebugControlPoints.ConnectPipelineProgramAndVBOAndShaderVariable(
        &debugProgram,
        &vboDebugControlPos,
        "position"
    );

    // --- Spline curve ---
    std::vector<float> splinePositions;

    for (int seg = 1; seg <= N - 3; ++seg)
    {
        for (int s = 0; s <= samplesPerSeg; ++s)
        {
            // avoid duplicating first sample of each later segment
            if (seg > 1 && s == 0)
                continue;

            float u = float(s) / float(samplesPerSeg);

            Point p = crEval(
                spline.points[seg - 1],
                spline.points[seg],
                spline.points[seg + 1],
                spline.points[seg + 2],
                u
            );

            splinePositions.push_back(p.x);
            splinePositions.push_back(p.y);
            splinePositions.push_back(p.z);
        }
    }

    debugSplineVertexCount = static_cast<int>(splinePositions.size() / 3);

    vboDebugSplinePos.Gen(
        debugSplineVertexCount,
        3,
        splinePositions.data(),
        GL_STATIC_DRAW
    );

    vaoDebugSpline.Gen();
    vaoDebugSpline.ConnectPipelineProgramAndVBOAndShaderVariable(
        &debugProgram,
        &vboDebugSplinePos,
        "position"
    );
}

void buildPillarGeometry(int samplesPerSeg = 100)
{
    const int N = spline.numControlPoints;
    if (N < 4) return;

    std::vector<float> positions;
    std::vector<float> normals;

    const Vec3 worldUp = { 0.0f, 1.0f, 0.0f };

    // Add a triangle with a flat face normal.
    auto emitTri = [&](Vec3 a, Vec3 b, Vec3 c)
    {
        Vec3 normal = nrm(cross(sub(b, a), sub(c, a)));

        for (Vec3 p : {a, b, c})
        {
            positions.insert(positions.end(),
                {p.x, p.y, p.z});

            normals.insert(normals.end(),
                {normal.x, normal.y, normal.z, 1.0f});
        }
    };

    auto emitQuad = [&](Vec3 a, Vec3 b, Vec3 c, Vec3 d)
    {
        emitTri(a, b, c);
        emitTri(c, b, d);
    };

    // Create a square beam between two arbitrary points.
    auto addBeam = [&](Vec3 start, Vec3 end)
    {
        Vec3 direction = nrm(sub(end, start));
        if (len(direction) < 1e-6f) return;

        // Construct a local frame around the beam.
        Vec3 reference = worldUp;

        if (std::abs(dot(direction, reference)) > 0.99f)
            reference = {1.0f, 0.0f, 0.0f};

        Vec3 right = nrm(cross(direction, reference));
        Vec3 up = nrm(cross(right, direction));

        float h = pillarWidth * 0.5f;

        Vec3 r = mul(right, h);
        Vec3 v = mul(up, h);

        // Four corners at each endpoint.
        Vec3 a0 = add(start, add(r, v));
        Vec3 a1 = add(start, sub(r, v));
        Vec3 a2 = sub(start, add(r, v));
        Vec3 a3 = add(start, sub(v, r));

        Vec3 b0 = add(end, add(r, v));
        Vec3 b1 = add(end, sub(r, v));
        Vec3 b2 = sub(end, add(r, v));
        Vec3 b3 = add(end, sub(v, r));

        // Four side faces.
        emitQuad(a0, b0, a1, b1);
        emitQuad(a1, b1, a2, b2);
        emitQuad(a2, b2, a3, b3);
        emitQuad(a3, b3, a0, b0);

        // End caps.
        emitTri(a0, a1, a2);
        emitTri(a0, a2, a3);

        emitTri(b0, b2, b1);
        emitTri(b0, b3, b2);
    };

    // Generate one A-frame at a spline position.
    auto addSupport = [&](Vec3 p, Vec3 tangent)
    {
        float topY = p.y - pillarTopOffset;

        if (topY <= pillarGroundY) return;

        // Horizontal right direction perpendicular to travel.
        Vec3 forward = nrm({tangent.x, 0.0f, tangent.z});

        if (len(forward) < 1e-6f)
            forward = {0.0f, 0.0f, -1.0f};

        Vec3 right = nrm(cross(forward, worldUp));

        // Apex directly below the track center.
        Vec3 top = {p.x, topY, p.z};

        // Two ground endpoints on opposite sides.
        Vec3 leftBase = add(
            Vec3{p.x, pillarGroundY, p.z},
            mul(right, -pillarSpread * 0.5f)
        );

        Vec3 rightBase = add(
            Vec3{p.x, pillarGroundY, p.z},
            mul(right, pillarSpread * 0.5f)
        );

        addBeam(top, leftBase);
        addBeam(top, rightBase);
    };

    // Sample spline and place supports at intervals.
    Vec3 previous = {0, 0, 0};
    bool firstPoint = true;
    float accumulatedDistance = 0.0f;
    int supportCount = 0;

    for (int seg = 1; seg <= N - 3; ++seg)
    {
        for (int s = 0; s <= samplesPerSeg; ++s)
        {
            if (seg > 1 && s == 0) continue;

            float u = float(s) / float(samplesPerSeg);

            const Point& Pm1 = spline.points[seg - 1];
            const Point& P0 = spline.points[seg];
            const Point& P1 = spline.points[seg + 1];
            const Point& P2 = spline.points[seg + 2];

            Point p = crEval(Pm1, P0, P1, P2, u);
            Point t = crTangent(Pm1, P0, P1, P2, u);

            Vec3 current = {p.x, p.y, p.z};
            Vec3 tangent = nrm({t.x, t.y, t.z});

            if (firstPoint)
            {
                addSupport(current, tangent);
                supportCount++;
                previous = current;
                firstPoint = false;
                continue;
            }

            accumulatedDistance += len(sub(current, previous));

            if (accumulatedDistance >= pillarSpacing)
            {
                addSupport(current, tangent);
                supportCount++;
                accumulatedDistance = 0.0f;
            }

            previous = current;
        }
    }

    // Upload geometry.
    pillarVertexCount = static_cast<int>(positions.size() / 3);

    if (pillarVertexCount == 0) return;

    vboPillarPos.Gen(
        pillarVertexCount, 3,
        positions.data(), GL_STATIC_DRAW
    );

    vboPillarNormals.Gen(
        pillarVertexCount, 4,
        normals.data(), GL_STATIC_DRAW
    );

    vaoPillars.Gen();

    vaoPillars.ConnectPipelineProgramAndVBOAndShaderVariable(
        &pipelineProgram, &vboPillarPos, "position"
    );

    vaoPillars.ConnectPipelineProgramAndVBOAndShaderVariable(
        &pipelineProgram, &vboPillarNormals, "normal"
    );

    std::cout << "Generated " << supportCount
              << " A-frame supports." << std::endl;
}

void displayFunc()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Advance along the spline.
    trackS += rideSpeed * 0.016f;

    int seg;
    float u;
    paramToSegU(trackS, seg, u, spline.numControlPoints);

    // Evaluate spline position and tangent.
    const Point& Pm1 = spline.points[seg - 1];
    const Point& P0  = spline.points[seg];
    const Point& P1  = spline.points[seg + 1];
    const Point& P2  = spline.points[seg + 2];

    Point p  = crEval(Pm1, P0, P1, P2, u);
    Point tp = crTangent(Pm1, P0, P1, P2, u);

    Vec3 P = { p.x, p.y, p.z };
    Vec3 T = nrm({ tp.x, tp.y, tp.z });

    // Update the spline frame for rail-relative information.
    Vec3 U = prevUp;
    Vec3 B;
    buildFrame(T, U, B);
    prevUp = U;

    // --- Camera position and orientation ---

    const Vec3 worldUp = { 0.0f, 1.0f, 0.0f };
    const float camHeight = 1.0f;
    const float degToRad = 3.14159265359f / 180.0f;

    // Follow spline position while remaining vertically offset.
    Vec3 eye = add(P, mul(worldUp, camHeight));

    // Preserve track pitch without inheriting its roll.
    Vec3 forward = T;

    float yaw = cameraYaw * degToRad;
    float pitch = cameraPitch * degToRad;

    // Apply yaw around world up.
    Vec3 horizontalRight = nrm(cross(forward, worldUp));

    if (len(horizontalRight) < 1e-6f)
        horizontalRight = { 1.0f, 0.0f, 0.0f };

    forward = nrm(add(
        mul(forward, cosf(yaw)),
        mul(horizontalRight, sinf(yaw))
    ));

    // Apply pitch relative to the camera.
    Vec3 right = nrm(cross(forward, worldUp));

    if (len(right) < 1e-6f)
        right = { 1.0f, 0.0f, 0.0f };

    Vec3 up = nrm(cross(right, forward));

    forward = nrm(add(
        mul(forward, cosf(pitch)),
        mul(up, sinf(pitch))
    ));

    // Reconstruct an orthogonal up vector to eliminate roll.
    right = nrm(cross(forward, worldUp));

    if (len(right) < 1e-6f)
        right = { 1.0f, 0.0f, 0.0f };

    Vec3 cameraUp = nrm(cross(right, forward));
    Vec3 center = add(eye, forward);

    // Construct the view matrix.
    matrix.SetMatrixMode(OpenGLMatrix::ModelView);
    matrix.LoadIdentity();

    matrix.LookAt(
        eye.x, eye.y, eye.z,
        center.x, center.y, center.z,
        cameraUp.x, cameraUp.y, cameraUp.z
    );

    // Preserve existing translation and scaling controls.
    float modelViewMatrix[16];

    matrix.Translate(terrainTranslate[0],
                     terrainTranslate[1],
                     terrainTranslate[2]);

    matrix.Translate(P.x, P.y, P.z);

    matrix.Scale(terrainScale[0],
                 terrainScale[1],
                 terrainScale[2]);

    matrix.Translate(-P.x, -P.y, -P.z);

    matrix.GetMatrix(modelViewMatrix);

    // Get projection matrix.
    float projectionMatrix[16];

    matrix.SetMatrixMode(OpenGLMatrix::Projection);
    matrix.GetMatrix(projectionMatrix);

    // --- Render roller coaster rail ---

    pipelineProgram.Bind();
    if (!showSplineDebug)
    {
      pipelineProgram.SetUniformVariableMatrix4fv(
          "modelViewMatrix", GL_FALSE, modelViewMatrix);

      pipelineProgram.SetUniformVariableMatrix4fv(
          "projectionMatrix", GL_FALSE, projectionMatrix);

      pipelineProgram.SetUniformVariable3f("lightDir", 0.4f, 0.8f, 0.4f);
      pipelineProgram.SetUniformVariable3f("ambientColor", 0.1f, 0.1f, 0.1f);
      pipelineProgram.SetUniformVariable3f("diffuseColor", 0.9f, 0.9f, 0.9f);
      pipelineProgram.SetUniformVariable3f("specularColor", 0.4f, 0.4f, 0.4f);
      pipelineProgram.SetUniformVariablef("shininess", 32.0f);

      glBindVertexArray(vaoSpline.GetHandle());
      glDrawArrays(GL_TRIANGLES, 0, splineVertexCount);

      // Draw support pillars using the same Phong shader.
      glBindVertexArray(vaoPillars.GetHandle());
      glDrawArrays(GL_TRIANGLES, 0, pillarVertexCount);
    }


    // --- Render textured ground ---

    textureProgram.Bind();

    textureProgram.SetUniformVariableMatrix4fv(
        "modelViewMatrix", GL_FALSE, modelViewMatrix);

    textureProgram.SetUniformVariableMatrix4fv(
        "projectionMatrix", GL_FALSE, projectionMatrix);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, groundTexture);

    textureProgram.SetUniformVariablei("tex", 0);

    glBindVertexArray(vaoGround.GetHandle());
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // --- Render skybox ---

    // Remove translation so the skybox stays centered on the camera.
    float skyboxViewMatrix[16];
    memcpy(skyboxViewMatrix, modelViewMatrix, sizeof(skyboxViewMatrix));

    skyboxViewMatrix[12] = 0.0f;
    skyboxViewMatrix[13] = 0.0f;
    skyboxViewMatrix[14] = 0.0f;

    // Render at far depth without modifying the depth buffer.
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);

    skyboxProgram.Bind();

    skyboxProgram.SetUniformVariableMatrix4fv(
        "viewMatrix", GL_FALSE, skyboxViewMatrix);

    skyboxProgram.SetUniformVariableMatrix4fv(
        "projectionMatrix", GL_FALSE, projectionMatrix);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);

    skyboxProgram.SetUniformVariablei("skyboxTexture", 0);

    glBindVertexArray(vaoSkybox.GetHandle());
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // Restore OpenGL state.
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);

    if (showSplineDebug)
    {
        debugProgram.Bind();

        debugProgram.SetUniformVariableMatrix4fv(
            "modelViewMatrix", GL_FALSE, modelViewMatrix);
        debugProgram.SetUniformVariableMatrix4fv(
            "projectionMatrix", GL_FALSE, projectionMatrix);

        // Draw spline curve
        debugProgram.SetUniformVariablef("pointSize", 1.0f);
        debugProgram.SetUniformVariable3f("color", 1.0f, 0.2f, 0.2f); // red

        glLineWidth(3.0f);
        glBindVertexArray(vaoDebugSpline.GetHandle());
        glDrawArrays(GL_LINE_STRIP, 0, debugSplineVertexCount);

        // Draw control points
        debugProgram.SetUniformVariablef("pointSize", 10.0f);
        debugProgram.SetUniformVariable3f("color", 0.2f, 1.0f, 0.2f); // green

        glBindVertexArray(vaoDebugControlPoints.GetHandle());
        glDrawArrays(GL_POINTS, 0, debugControlPointCount);
    }

    glutSwapBuffers();
}


//make different vertex arrays for each mode
void geometryForMode() {
    const float heightScale = 0.1f;
    //center the height map
    const float sx = (imageW > 1) ? 1.0f / (imageW - 1) : 1.0f;
    const float sy = (imageH > 1) ? 1.0f / (imageH - 1) : 1.0f;

    //"1" - points mode
    if (rm == RenderMode::Points) {
        numVertices = imageW * imageH; //one vertex per pixel

        std::unique_ptr<float[]> positions = std::make_unique<float[]>(numVertices * 3); //positions
        std::unique_ptr<float[]> colors = std::make_unique<float[]>(numVertices * 4); //colors

        for (int y = 0; y < imageH; ++y) {
            for (int x = 0; x < imageW; ++x) {
                int ind = (y * imageW + x) * 3;
                const float pixelHeight = pixels[y * imageW + x] / 255.0f;
                positions[ind] = x * sx - 0.5f;
                positions[ind + 1] = pixelHeight * heightScale;
                positions[ind + 2] = y * sy - 0.5f;

                ind = (y * imageW + x) * 4;
                colors[ind] = 1.0f; colors[ind + 1] = 1.0f; colors[ind + 2] = 1.0f;
                colors[ind + 3] = pixelHeight;
            }
        }

        vboVertices.Gen(numVertices, 3, positions.get(), GL_STATIC_DRAW); // 3 values per position
        vboColors.Gen(numVertices, 4, colors.get(), GL_STATIC_DRAW); // 4 values per color
    }

    //"2" - grid mode
    else if (rm == RenderMode::Grid) {
        numVertices = imageW * (imageW * 4 - 4); //derived this

        std::unique_ptr<float[]> positions = std::make_unique<float[]>(numVertices * 3); //positions
        std::unique_ptr<float[]> colors = std::make_unique<float[]>(numVertices * 4); //colors

        int ind = 0;
        //horizontal lines
        for (int y = 0; y < imageH; ++y) {
            for (int x = 0; x < imageW - 1; ++x) {
                float pixelHeightA = pixels[y * imageW + x] / 255.0f;
                positions[ind * 3] = x * sx - 0.5f;
                positions[ind * 3 + 1] = pixelHeightA * heightScale;
                positions[ind * 3 + 2] = y * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = pixelHeightA; 
                ++ind;

                float pixelHeightB = pixels[y * imageW + (x + 1)] / 255.0f;
                positions[ind * 3] = (x + 1) * sx - 0.5f;
                positions[ind * 3 + 1] = pixelHeightB * heightScale;
                positions[ind * 3 + 2] = y * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = pixelHeightB;
                ++ind;
            }
        }

        //vertical lines
        for (int x = 0; x < imageW; ++x) {
            for (int y = 0; y < imageH - 1; ++y) {
                float pixelHeightA = pixels[y * imageW + x] / 255.0f;
                positions[ind * 3] = x * sx - 0.5f;
                positions[ind * 3 + 1] = pixelHeightA * heightScale;
                positions[ind * 3 + 2] = y * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = pixelHeightA; 
                ++ind;

                float pixelHeightB = pixels[(y + 1) * imageW + x] / 255.0f;
                positions[ind * 3] = x * sx - 0.5f;
                positions[ind * 3 + 1] = pixelHeightB * heightScale;
                positions[ind * 3 + 2] = (y + 1) * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = pixelHeightB;
                ++ind;
            }
        }

        vboVertices.Gen(numVertices, 3, positions.get(), GL_STATIC_DRAW); // 3 values per position
        vboColors.Gen(numVertices, 4, colors.get(), GL_STATIC_DRAW); // 4 values per color
    }

    //"3" - triangles mode
    else if (rm == RenderMode::Tris) {
        numVertices = 6 * (imageW - 1) * (imageH - 1); //each square has 6 vertices

        std::unique_ptr<float[]> positions = std::make_unique<float[]>(numVertices * 3); //positions
        std::unique_ptr<float[]> colors = std::make_unique<float[]>(numVertices * 4); //colors

        int ind = 0;
        for (int y = 0; y < imageH - 1; ++y) {
            for (int x = 0; x < imageW - 1; ++x) {
                float height00 = pixels[y * imageW + x] / 255.0f;
                float height10 = pixels[y * imageW + (x + 1)] / 255.0f;
                float height01 = pixels[(y + 1) * imageW + x] / 255.0f;
                float height11 = pixels[(y + 1) * imageW + (x + 1)] / 255.0f;

                //top right? tri
                positions[ind * 3] = x * sx - 0.5f;
                positions[ind * 3 + 1] = height00 * heightScale;
                positions[ind * 3 + 2] = y * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = height00; 
                ++ind;

                positions[ind * 3] = (x + 1) * sx - 0.5f;
                positions[ind * 3 + 1] = height10 * heightScale;
                positions[ind * 3 + 2] = y * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = height10;
                ++ind;

                positions[ind * 3] = (x + 1) * sx - 0.5f;
                positions[ind * 3 + 1] = height11 * heightScale;
                positions[ind * 3 + 2] = (y + 1) * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = height11;
                ++ind;

                //bot left? tri
                positions[ind * 3] = x * sx - 0.5f;
                positions[ind * 3 + 1] = height00 * heightScale;
                positions[ind * 3 + 2] = y * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = height00;
                ++ind;

                positions[ind * 3] = (x + 1) * sx - 0.5f;
                positions[ind * 3 + 1] = height11 * heightScale;
                positions[ind * 3 + 2] = (y + 1) * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = height11;
                ++ind;

                positions[ind * 3] = x * sx - 0.5f;
                positions[ind * 3 + 1] = height01 * heightScale;
                positions[ind * 3 + 2] = (y + 1) * sy - 0.5f;
                colors[ind * 4] = 1.0f; colors[ind * 4 + 1] = 1.0f;  colors[ind * 4 + 2] = 1.0f;
                colors[ind * 4 + 3] = height01;
                ++ind;
            }
        }
        vboVertices.Gen(numVertices, 3, positions.get(), GL_STATIC_DRAW); // 3 values per position
        vboColors.Gen(numVertices, 4, colors.get(), GL_STATIC_DRAW); // 4 values per color
    }

    vao.Gen();

    //"4" - smooth mode
    if (rm == RenderMode::Smooth) {
        numVertices = 6 * (imageW - 1) * (imageH - 1);

        std::unique_ptr<float[]> positions = std::make_unique<float[]>(numVertices * 3); //positions
        std::unique_ptr<float[]> colors = std::make_unique<float[]>(numVertices * 4); //colors

        //additional vbos for smoothing
        std::unique_ptr<float[]> pLeft = std::make_unique<float[]>(numVertices * 3);
        std::unique_ptr<float[]> pRight = std::make_unique<float[]>(numVertices * 3);
        std::unique_ptr<float[]> pUp = std::make_unique<float[]>(numVertices * 3);
        std::unique_ptr<float[]> pDown = std::make_unique<float[]>(numVertices * 3);

        //helper function to write self and neighbors
        auto smoothHelper = [&](int ind, int x, int y) {
            float pixelHeight = pixels[y * imageW + x] / 255.0f;
            positions[ind * 3] = x * sx - 0.5f;
            positions[ind * 3 + 1] = pixelHeight;
            positions[ind * 3 + 2] = y * sy - 0.5f;

            colors[ind * 4 + 0] = 1.0f; colors[ind * 4 + 1] = 1.0f; colors[ind * 4 + 2] = 1.0f; 
            colors[ind * 4 + 3] = pixelHeight;

            //bounds check - if its an edge vertex, use self
            int leftX = (x > 0) ? x - 1 : x;
            int rightX = (x < imageW - 1) ? x + 1 : x;
            int downY = (y > 0) ? y - 1 : y;
            int upY = (y < imageH - 1) ? y + 1 : y;

            auto setNeighbors = [&](float* arr, int newX, int newY) {
                float newHeight = pixels[newY * imageW + newX] / 255.0f;
                arr[ind * 3 + 0] = newX * sx - 0.5f;
                arr[ind * 3 + 1] = newHeight;                 
                arr[ind * 3 + 2] = newY * sy - 0.5f;
                };

            setNeighbors(pLeft.get(), leftX, y);
            setNeighbors(pRight.get(), rightX, y);
            setNeighbors(pDown.get(), x, downY);
            setNeighbors(pUp.get(), x, upY);
            };

        int ind = 0;
        for (int y = 0; y < imageH - 1; ++y) {
            for (int x = 0; x < imageW - 1; ++x) {
                //tri 1
                smoothHelper(ind++, x, y);
                smoothHelper(ind++, x + 1, y);
                smoothHelper(ind++, x + 1, y + 1);

                //tri 2
                smoothHelper(ind++, x, y);
                smoothHelper(ind++, x + 1, y + 1);
                smoothHelper(ind++, x, y + 1);
            }
        }

        vboVertices.Gen(numVertices, 3, positions.get(), GL_STATIC_DRAW);
        vboColors.Gen(numVertices, 4, colors.get(), GL_STATIC_DRAW); 
        vboLeft.Gen(numVertices, 3, pLeft.get(), GL_STATIC_DRAW); 
        vboRight.Gen(numVertices, 3, pRight.get(), GL_STATIC_DRAW); 
        vboUp.Gen(numVertices, 3, pUp.get(), GL_STATIC_DRAW); 
        vboDown.Gen(numVertices, 3, pDown.get(), GL_STATIC_DRAW);

        vao.ConnectPipelineProgramAndVBOAndShaderVariable(&pipelineProgram, &vboLeft, "pLeft");
        vao.ConnectPipelineProgramAndVBOAndShaderVariable(&pipelineProgram, &vboRight, "pRight");
        vao.ConnectPipelineProgramAndVBOAndShaderVariable(&pipelineProgram, &vboUp, "pUp");
        vao.ConnectPipelineProgramAndVBOAndShaderVariable(&pipelineProgram, &vboDown, "pDown");
    }

    vao.ConnectPipelineProgramAndVBOAndShaderVariable(&pipelineProgram, &vboVertices, "position");
    vao.ConnectPipelineProgramAndVBOAndShaderVariable(&pipelineProgram, &vboColors, "color");
}

bool loadCubemap(const char* faces[6])
{
    glGenTextures(1, &skyboxTexture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);

    int expectedSize = -1;

    for (int i = 0; i < 6; ++i)
    {
        ImageIO img;
        ImageIO::fileFormatType format;

        if (img.load(faces[i], &format) != ImageIO::OK)
        {
            std::cerr << "Failed to load skybox face: "
                      << faces[i] << std::endl;
            return false;
        }

        int width = img.getWidth();
        int height = img.getHeight();
        int channels = img.getBytesPerPixel();

        if (width != height ||
            (expectedSize != -1 && width != expectedSize))
        {
            std::cerr << "Skybox faces must be equal-sized squares."
                      << std::endl;
            return false;
        }

        if (channels < 1 || channels > 4)
        {
            std::cerr << "Unsupported skybox image format."
                      << std::endl;
            return false;
        }

        expectedSize = width;

        std::vector<unsigned char> pixels(width * height * 3);

        // Cubemap face order:
        // 0 = Right (+X)
        // 1 = Left  (-X)
        // 2 = Up    (+Y)
        // 3 = Down  (-Y)
        // 4 = Front (+Z)
        // 5 = Back  (-Z)

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int sourceX = x;
                int sourceY = y;

                // Correct image orientation for each face.
                switch (i)
                {
                    case 0: // Right
                    case 1: // Left
                    case 4: // Front
                    case 5: // Back
                        sourceX = width - 1 - x;
                        sourceY = height - 1 - y;
                        break;

                    case 2: // Up: rotate 90 degrees
                        sourceX = y;
                        sourceY = height - 1 - x;
                        break;

                    case 3: // Down: no adjustment
                    default:
                        break;
                }

                int dst = (y * width + x) * 3;

                for (int c = 0; c < 3; ++c)
                {
                    int srcChannel = (channels >= 3) ? c : 0;

                    pixels[dst + c] =
                        img.getPixel(sourceX, sourceY, srcChannel);
                }
            }
        }

        // Upload the current face to the GPU.
        glTexImage2D(
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
            0,
            GL_RGB8,
            width,
            height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            pixels.data()
        );

        std::cout << "Loaded skybox face: "
                  << faces[i] << std::endl;
    }

    // Set cubemap texture parameters.
    glTexParameteri(GL_TEXTURE_CUBE_MAP,
                    GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP,
                    GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_CUBE_MAP,
                    GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP,
                    GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP,
                    GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    return true;
}


void buildSkyboxGeometry()
{
    const float vertices[] = {
        //back
        -1,  1, -1,  -1, -1, -1,   1, -1, -1,
         1, -1, -1,   1,  1, -1,  -1,  1, -1,

        //left
        -1, -1,  1,  -1, -1, -1,  -1,  1, -1,
        -1,  1, -1,  -1,  1,  1,  -1, -1,  1,

        //right
         1, -1, -1,   1, -1,  1,   1,  1,  1,
         1,  1,  1,   1,  1, -1,   1, -1, -1,

        //front
        -1, -1,  1,  -1,  1,  1,   1,  1,  1,
         1,  1,  1,   1, -1,  1,  -1, -1,  1,

        //top
        -1,  1, -1,   1,  1, -1,   1,  1,  1,
         1,  1,  1,  -1,  1,  1,  -1,  1, -1,

        //bottom
        -1, -1, -1,  -1, -1,  1,   1, -1, -1,
         1, -1, -1,  -1, -1,  1,   1, -1,  1
    };

    vboSkyboxPos.Gen(36, 3, vertices, GL_STATIC_DRAW);

    vaoSkybox.Gen();
    vaoSkybox.ConnectPipelineProgramAndVBOAndShaderVariable(
        &skyboxProgram,
        &vboSkyboxPos,
        "position"
    );
}

void initScene(int argc, char *argv[])
{
  // Set the background color.
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black color.

  // Enable z-buffering (i.e., hidden surface removal using the z-buffer algorithm).
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Create a pipeline program. This operation must be performed BEFORE we initialize any VAOs.
  // A pipeline program contains our shaders. Different pipeline programs may contain different shaders.
  // In this homework, we only have one set of shaders, and therefore, there is only one pipeline program.
  // In hw2, we will need to shade different objects with different shaders, and therefore, we will have
  // several pipeline programs (e.g., one for the rails, one for the ground/sky, etc.).
  // Load and set up the pipeline program, including its shaders.
  if (pipelineProgram.BuildShadersFromFiles(shaderBasePath, "vertexShader.glsl", "fragmentShader.glsl") != 0)
  {
    cout << "Failed to build the pipeline program." << endl;
    throw 1;
  } 
  cout << "Successfully built the pipeline program." << endl;
  
  // build texture shader pipeline for ground
  if (textureProgram.BuildShadersFromFiles(shaderBasePath, "textureVertexShader.glsl", "textureFragmentShader.glsl") != 0)
  {
    cout << "Failed to build the texture pipeline program." << endl;
    throw 1;
  }
  cout << "Successfully built the texture pipeline program." << endl;
    

  // initialize skybox shaders
  if (skyboxProgram.BuildShadersFromFiles(
      shaderBasePath,
      "skyboxVertexShader.glsl",
      "skyboxFragmentShader.glsl") != 0)
  {
      std::cerr << "Failed to build skybox shaders." << std::endl;
      throw 1;
  }

  std::cout << "Successfully built skybox shaders." << std::endl;

  if (debugProgram.BuildShadersFromFiles(
        shaderBasePath,
        "debugVertexShader.glsl",
        "debugFragmentShader.glsl") != 0)
  {
      std::cout << "Failed to build debug pipeline program." << std::endl;
      throw 1;
  }
  std::cout << "Successfully built debug pipeline program." << std::endl;

  // Bind the pipeline program that we just created. 
  // The purpose of binding a pipeline program is to activate the shaders that it contains, i.e.,
  // any object rendered from that point on, will use those shaders.
  // When the application starts, no pipeline program is bound, which means that rendering is not set up.
  // So, at some point (such as below), we need to bind a pipeline program.
  // From that point on, exactly one pipeline program is bound at any moment of time.
  // Check for any OpenGL errors.
  std::cout << "GL error status is: " << glGetError() << std::endl;
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
      printf("Usage: %s <spline file>\n", argv[0]);
      exit(0);
  }

  // load spline
  loadSpline(argv[1]);

  printf("Loaded spline with %d control point(s).\n", spline.numControlPoints);

  cout << "Initializing GLUT..." << endl;
  glutInit(&argc,argv);

  cout << "Initializing OpenGL..." << endl;

  #ifdef __APPLE__
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #else
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #endif

  glutInitWindowSize(windowWidth, windowHeight);
  glutInitWindowPosition(0, 0);  
  glutCreateWindow(windowTitle);

  cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
  cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
  cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

  #ifdef __APPLE__
    // This is needed on recent Mac OS X versions to correctly display the window.
    glutReshapeWindow(windowWidth - 1, windowHeight - 1);
  #endif

  // Tells GLUT to use a particular display function to redraw.
  glutDisplayFunc(displayFunc);
  // Perform animation inside idleFunc.
  glutIdleFunc(idleFunc);
  // callback for mouse drags
  glutMotionFunc(mouseMotionDragFunc);
  // callback for idle mouse movement
  glutPassiveMotionFunc(mouseMotionFunc);
  // callback for mouse button changes
  glutMouseFunc(mouseButtonFunc);
  // callback for resizing the window
  glutReshapeFunc(reshapeFunc);
  // callback for pressing the keys on the keyboard
  glutKeyboardFunc(keyboardFunc);

  // init glew
  #ifdef __APPLE__
    // nothing is needed on Apple
  #else
    // Windows, Linux
    GLint result = glewInit();
    if (result != GLEW_OK)
    {
      cout << "error: " << glewGetErrorString(result) << endl;
      exit(EXIT_FAILURE);
    }
  #endif

  // Perform the initialization.
  initScene(argc, argv);
  //pipelineProgram.Bind();
  
  buildSkyboxGeometry();
  buildDebugSplineGeometry(200);

  const char* skyboxFaces[6] = {
      "textures/skybox/right.jpg",
      "textures/skybox/left.jpg",
      "textures/skybox/up.jpg",
      "textures/skybox/down.jpg",
      "textures/skybox/front.jpg",
      "textures/skybox/back.jpg"
  };

  if (!loadCubemap(skyboxFaces))
  {
      std::cerr << "Could not load cubemap skybox." << std::endl;
      return 1;
  }

  buildSplineGeometry(500);
  buildPillarGeometry(100);
  
  // Build a large ground plane below the track and load texture
  {
    const float y = -1.0f;
    const float half = 200.0f;
    const float tile = 50.0f; // how many times the texture repeats

    float groundPos[] = {
      -half, y, -half,
       half, y, -half,
      -half, y,  half,
      -half, y,  half,
       half, y, -half,
       half, y,  half
    };

    float groundUV[] = {
      0.0f, 0.0f,
      tile, 0.0f,
      0.0f, tile,
      0.0f, tile,
      tile, 0.0f,
      tile, tile
    };

    vboGroundPos.Gen(6, 3, groundPos, GL_STATIC_DRAW);
    vboGroundTex.Gen(6, 2, groundUV, GL_STATIC_DRAW);
    vaoGround.Gen();
    vaoGround.ConnectPipelineProgramAndVBOAndShaderVariable(&textureProgram, &vboGroundPos, "position");
    vaoGround.ConnectPipelineProgramAndVBOAndShaderVariable(&textureProgram, &vboGroundTex, "texCoord");

    glGenTextures(1, &groundTexture);
    // robust built-in texture path relative to Bin/Debug -> up to hw1 -> heightmap
    initTexture("textures/rocky_terrain_02_diff_1k.jpg", groundTexture);
  }

  // Sink forever into the GLUT loop.
  glutMainLoop();
  return 0;
}