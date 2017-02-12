#version 330

uniform sampler2D texture;

in mediump vec2 texCoords;

layout(location = 0) out highp vec4 colour;

void main()
{
	colour = texture2D(texture, texCoords);
	colour.g += colour.r;
	colour.b += colour.r;
}