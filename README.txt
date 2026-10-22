# Procedural Roller Coaster | OpenGL

A real-time roller coaster simulation built in **C++ and OpenGL**, featuring procedurally generated track geometry, spline-based camera movement, and custom shader rendering.

## Features

- **Catmull–Rom Spline Interpolation:** Constructs a continuous 3D path from control points using cubic interpolation.
- **Procedural Track Generation:** Generates T-shaped rail geometry using spline positions, tangent vectors, and local coordinate frames.
- **Procedural Support Structures:** Generates A-frame supports at regular intervals along the track, adapting to its elevation and direction.
- **Spline-Following Camera:** Animates the camera along the track using interpolated positions and tangent-based orientation.
- **Interactive Camera Controls:** Supports mouse-controlled yaw and pitch with constrained pitch and locked camera roll.
- **OpenGL Rendering:** Implements Phong shading, texture mapping, and cubemap skybox rendering using GLSL shaders.

## Technologies

- C++14
- OpenGL 3.2+ (Core Profile)
- GLSL
- GLUT
- Catmull–Rom Splines

## Build & Run

### macOS

Requires a C++ compiler, OpenGL/GLUT frameworks, and `jpeg-turbo`.

Run the simulation with a spline file:

```bash
./hw1 ../splines/goodRide.sp
```

### Controls

| Input | Action |
|---|---|
| Left-click + drag | Look around (yaw/pitch) |
| P | Toggle spline/control point visualization |
| X | Save screenshot |
| Esc | Exit |

## Author

Anna Zhu — USC CSCI 420: Computer Graphics