// vertex
#version 330 core

uniform mat4x4 view;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 textureCoord;
layout (location = 2) in vec4 color;

out vec2 TexCoord;
out vec4 Color;
out vec2 FragPosition;

void main()
{
    gl_Position = view * vec4(aPos, 1.0);
    TexCoord = textureCoord;
    Color = color;
    FragPosition = aPos.xy;
}

// fragment
#version 330 core

uniform sampler2D spriteTexture;

in vec2 TexCoord;
in vec4 Color;
in vec2 FragPosition;

out vec4 FragColor;

void main()
{
    vec4 texel = texture(spriteTexture, TexCoord);

    if(texel.a == 0) discard;

    FragColor = vec4(mix(texel.rgb, Color.rgb, min(Color.a, 1)), texel.a);
}