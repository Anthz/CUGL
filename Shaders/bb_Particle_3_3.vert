#version 330

in highp vec3 aPos;
in highp vec4 iPos;

uniform highp mat4 uViewMatrix;
uniform highp mat4 uProjMatrix;

void main()
{
	vec2 bbSize = vec2(5e9, 5e9);
	vec3 bbPos;
	bbPos = iPos.xyz + vec3(uViewMatrix[0][0], uViewMatrix[1][0], uViewMatrix[2][0])
			* aPos.x * bbSize.x + vec3(uViewMatrix[0][1], uViewMatrix[1][1], uViewMatrix[2][1])
           * aPos.y * bbSize.y;
	gl_Position = uProjMatrix * uViewMatrix * vec4(bbPos, 1.0);
}