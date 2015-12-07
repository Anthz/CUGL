#version 330

in highp vec3 aPos;
in mediump vec2 aUV;

out mediump vec2 texCoords;

void main()
{
   texCoords = aUV;
   gl_Position = vec4(aPos, 1.0);
}