attribute highp vec3 aPos;
attribute highp vec3 iPos;
attribute mediump vec2 aUV;
uniform highp mat4 uModelMatrix;
uniform highp mat4 uViewMatrix;
uniform highp mat4 uProjMatrix;
varying mediump vec2 texCoords;
void main() {
   texCoords = aUV;
   gl_Position = uProjMatrix * uViewMatrix * uModelMatrix * vec4(aPos + iPos, 1.0);
};