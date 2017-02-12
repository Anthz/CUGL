#version 330

uniform highp mat4 uModelMatrix;
uniform highp mat4 uViewMatrix;
uniform highp mat4 uProjMatrix;

in highp vec3 aPos;
in int gl_InstanceID;

void main()
{
   vec3 iPos = vec3(gl_InstanceID % 256, 0, gl_InstanceID / 256);
   iPos *= 2.0f; //2 = cell width
   gl_Position = uProjMatrix * uViewMatrix * uModelMatrix * vec4(aPos + iPos, 1.0);
}