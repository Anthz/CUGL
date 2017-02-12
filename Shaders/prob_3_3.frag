#version 330

uniform sampler2D tex;

layout(location = 0) out highp vec4 colour;
in mediump vec2 texCoords;

void main()
{
	colour = texture2D(tex, texCoords);
	
	if(colour == vec4(1.0, 1.0, 1.0, 1.0))
		colour = vec4(1.0, 0.0, 0.0, 1.0);
}