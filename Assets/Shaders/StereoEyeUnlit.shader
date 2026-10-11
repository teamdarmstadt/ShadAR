Shader "PromptedVision/StereoEyeUnlit"
{
    // Unlit textured quad shader with per-eye gating. URP ignores Camera.stereoTargetEye,
    // so the per-eye passthrough quads (QuadLeft/QuadRight) are rendered by ONE stereo
    // camera and each quad collapses itself in the eye it does not belong to.
    // _EyeIndex: -1 = visible in both eyes, 0 = left eye only, 1 = right eye only.
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _EyeIndex ("Eye Index (-1 both, 0 left, 1 right)", Float) = -1
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        Cull Off ZWrite On

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile _ UNITY_SINGLE_PASS_STEREO
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _EyeIndex;

            // unity_StereoEyeIndex only exists in stereo variants; non-stereo (editor) is eye 0.
            #if defined(USING_STEREO_MATRICES) || defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_STEREO_MULTIVIEW_ENABLED)
                #define PV_EYE_INDEX ((float)unity_StereoEyeIndex)
            #else
                #define PV_EYE_INDEX 0.0
            #endif

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; UNITY_VERTEX_INPUT_INSTANCE_ID };
            struct v2f { float2 uv : TEXCOORD0; float4 pos : SV_POSITION; UNITY_VERTEX_OUTPUT_STEREO };

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);

                // Collapse to a degenerate triangle when rendered for the other eye.
                if (_EyeIndex >= 0.0 && abs(PV_EYE_INDEX - _EyeIndex) > 0.5)
                {
                    o.pos = float4(0, 0, 0, 0);
                }
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);
                col.a = 1.0;
                return col;
            }
            ENDCG
        }
    }
    FallBack Off
}
