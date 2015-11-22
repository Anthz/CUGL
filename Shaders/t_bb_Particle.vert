attribute highp vec3 aPos;
attribute highp vec3 iPos;
attribute mediump vec2 aUV;

uniform highp mat4 uViewMatrix;
uniform highp mat4 uProjMatrix;

varying mediump vec2 texCoords;

void main() {
	texCoords = aUV;
	
	vec2 bbSize = vec2(1.0f, 1.0f);
	vec3 bbPos;
	bbPos = iPos + vec3(uViewMatrix[0][0], uViewMatrix[1][0], uViewMatrix[2][0])
			* aPos.x * bbSize.x + vec3(uViewMatrix[0][1], uViewMatrix[1][1], uViewMatrix[2][1])
           * aPos.y * bbSize.y;
	gl_Position = uProjMatrix * uViewMatrix * vec4(bbPos, 1.0);
};