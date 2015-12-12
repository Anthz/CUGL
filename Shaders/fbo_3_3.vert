#version 330

in highp vec3 aPos;
in mediump vec2 aUV;

out mediump vec2 texCoords;

void main()
{
   texCoords = vec2(aUV.x, 1 - aUV.y);
   gl_Position = vec4(aPos * 2.0, 1.0);
}