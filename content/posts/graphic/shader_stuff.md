+++
title = "Shader_stuff"
date = "2023-06-04T12:37:13+08:00"
author = "ikws4"
cover = ""
tags = []
keywords = []
readingTime = false
Toc = false
+++

<!--more-->

### Metaball

```glsl
#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

vec3 getMaterial(int matId) {
    if (matId == 1) return vec3(0.624,1.000,0.459);
    if (matId == 2) return vec3(0.926,1.000,0.301);
    if (matId == 3) return vec3(1.000,0.236,0.782);
    return vec3(0.);
}

vec4 metaball(vec2 p, float r, int matId) {
    float d = r / length(p);
    return vec4(getMaterial(matId) * d, d);
}

vec4 metasquare(vec2 p, float r, int matId) {
    vec2 q = abs(p) - vec2(r);
    float d = r / (length(max(q, 0.0)) + min(max(q.x, q.y),0.0) + r);
    return vec4(getMaterial(matId) * d, d);
}
 
#define SCALE 1.5
#define SHAPES_ARRAY_LENGTH 3

vec3 render(vec2 st) {
    vec2 mouse = ((u_mouse / u_resolution) - vec2(0.5)) * SCALE;
    
    vec4 shapes[SHAPES_ARRAY_LENGTH];
    shapes[0] = vec4(metaball(mouse - st, 0.112 * (pow(sin(u_time), 2.) * 0.5 + 1.), 1));
    shapes[1] = vec4(metaball(vec2(sin(u_time), cos(u_time)) * 0.2 - st, 0.03 * (pow(sin(u_time), 2.) * 0.5 + 1.), 2));
    shapes[2] = vec4(metaball(vec2(sin(u_time) * sin(u_time), cos(u_time) * sin(u_time)) * 0.1 - st, 0.03 * (pow(cos(u_time), 2.) * 0.5 + 1.), 3));
    
    float W = 0.;
    vec3 col = vec3(0.);
    for (int i = 0; i < SHAPES_ARRAY_LENGTH; i++) {
        W += shapes[i].w;
        col += shapes[i].rgb;
    }
    col = (col / W) * (W > 1. ? 1. : W * 0.380);
    
    return col.rgb;
}


void main() {
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    st -= vec2(0.5);
    st *= SCALE;
    
    vec3 col = render(st);
    gl_FragColor = vec4(vec3(col), 1.);
}
```
