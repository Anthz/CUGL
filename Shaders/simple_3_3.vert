#version 330

in highp vec3 aPos;
in highp vec4 iPos;
in mediump vec2 aUV;

uniform highp mat4 uModelMatrix;
uniform highp mat4 uViewMatrix;
uniform highp mat4 uProjMatrix;

out mediump vec2 texCoords;

void main()
{
   texCoords = aUV;
   gl_Position = uProjMatrix * uViewMatrix * uModelMatrix * vec4(aPos + iPos.xyz, 1.0);
}