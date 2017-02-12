#version 330

uniform highp mat4 uModelMatrix;
uniform highp mat4 uViewMatrix;
uniform highp mat4 uProjMatrix;
uniform sampler2D texture;

in highp vec3 aPos;
in highp vec4 iPos;
in mediump vec2 aUV;

out mediump vec2 texCoords;

void main()
{
   vec4 colour = texture2D(texture, aUV);
   texCoords = aUV;
   
   vec4 position = vec4(aPos + iPos.xyz, 1.0);
   position.y += colour.r;
   gl_Position = uProjMatrix * uViewMatrix * uModelMatrix * position;
}