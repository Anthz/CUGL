#version 330

uniform sampler2D texture;

in mediump vec2 texCoords;

out highp vec4 colour;

void main()
{
	colour = texture2D(texture, texCoords);
}