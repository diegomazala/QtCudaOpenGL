#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
#endif

uniform mat4 mvp_matrix;

attribute vec3 in_position;


void main()
{
    // Calculate vertex position in screen space
    gl_Position = mvp_matrix * vec4(in_position, 1);
}

