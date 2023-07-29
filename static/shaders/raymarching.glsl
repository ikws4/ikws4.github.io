#define SCREEN_SPACE_TO_CLIP_SPACE(pos) \
  ((pos / iResolution.xy) * vec2(iResolution.x / iResolution.y, 1.) - vec2(0.5)) * 5.

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
 
#define SHAPES_ARRAY_LENGTH 3

vec3 render(vec2 uv) {
    vec2 mouse = SCREEN_SPACE_TO_CLIP_SPACE(iMouse.xy);
    
    vec4 shapes[SHAPES_ARRAY_LENGTH];
    shapes[0] = vec4(metaball(mouse - uv, 0.112 * (pow(sin(iTime), 2.) * 0.5 + 1.), 1));
    shapes[1] = vec4(metasquare(vec2(sin(iTime), cos(iTime)) * 0.2 - uv, 0.03 * (pow(sin(iTime), 2.) * 0.5 + 1.), 2));
    shapes[2] = vec4(metasquare(vec2(sin(iTime) * sin(iTime), cos(iTime) * sin(iTime)) * 0.1 - uv, 0.03 * (pow(cos(iTime), 2.) * 0.5 + 1.), 3));
    
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
    vec2 uv = SCREEN_SPACE_TO_CLIP_SPACE(gl_FragCoord.xy);
    
    vec3 col = render(uv);
    gl_FragColor = vec4(vec3(col), 1.);
}