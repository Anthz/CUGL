uniform sampler2D texture;

varying mediump vec2 texCoords;

void main() {
	gl_FragColor = texture2D(texture, texCoords);
}