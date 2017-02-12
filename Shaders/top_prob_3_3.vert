#version 330

uniform highp mat4 uModelMatrix;
uniform highp mat4 uViewMatrix;
uniform highp mat4 uProjMatrix;

in highp vec3 aPos;
in mediump vec2 aUV;
in float tProb;
in float bProb;
in int gl_InstanceID;

out mediump vec2 texCoords;

void main()
{
   texCoords = aUV;
   
   vec3 iPos = vec3(gl_InstanceID % 256, 0, gl_InstanceID / 256);
   iPos *= 2.0f; //2 = cell width
   
   vec4 gridPos = vec4(aPos + iPos, 1.0);
   
   if(gridPos.y > 0)
   {
      gridPos.y *= (50 * tProb);
	  //texCoords.y *= (50 * tProb);
   }
   
   gl_Position = uProjMatrix * uViewMatrix * uModelMatrix * gridPos;
}