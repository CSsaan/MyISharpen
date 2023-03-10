varying vec2 vv2_Texcoord;   // 纹理坐标
uniform sampler2D m_texture; // 原图片

vec3 mtexSample(const float x, const float y)
{
    vec2 uv = vv2_Texcoord + vec2(x / 1280.0, y / 720.0); // 纹理分辨率：1280,720
    lowp vec3 textureColor = texture2D(m_texture, uv);    // 图像纹理采样
    return textureColor;
}

vec3 sharpen(vec2 fragCoord, float strength)
{
    //卷积核 (以拉普拉斯算子为例)
    vec3 f =
        mtexSample(-1.0, -1.0) * -1.0 +
        mtexSample(0.0, -1.0) * -1.0 +
        mtexSample(1.0, -1.0) * -1.0 +

        mtexSample(-1.0, 0.0) * -1.0 +
        mtexSample(0.0, 0.0) * 9.0 +
        mtexSample(1.0, 0.0) * -1.0 +

        mtexSample(-1.0, 1.0) * -1.0 +
        mtexSample(0.0, 1.0) * -1.0 +
        mtexSample(1.0, 1.0) * -1.0;

    return mix(vec4(mtexSample(0.0, 0.0), 1.0), vec4(f, 1.0), strength).rgb;
}

void main()
{
    vec3 sharpened = sharpen(vv2_Texcoord, 1.0);
    gl_FragColor = vec4(sharpened, 1.0);
}
