using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
/// Renders small "live" previews of history shaders for the grid menu.
///
/// Each slot holds a low-res RenderTexture and a private copy of the shader's material.
/// Every tick a few slots (round-robin) are refreshed by blitting the current camera
/// feed through their material — so the whole grid looks live at a fraction of the cost
/// of rendering every cell every frame.
/// </summary>
public class ShaderPreviewRenderer : IDisposable
{
    private static readonly int EyeIndexProp = Shader.PropertyToID("_EyeIndex");
    private static readonly int BaseMapProp = Shader.PropertyToID("_BaseMap");
    private static readonly int DetectionsProp = Shader.PropertyToID("_Detections");
    private static readonly int DetectionCountProp = Shader.PropertyToID("_DetectionCount");
    private static readonly int HasMaskProp = Shader.PropertyToID("_HasMask");
    private static readonly int MaskCountProp = Shader.PropertyToID("_MaskCount");

    private readonly int width;
    private int height;
    private readonly List<RenderTexture> rts = new();
    private readonly List<Material> mats = new();
    private readonly List<Material> sourceMats = new();
    private ComputeBuffer dummyDetectionBuffer;
    private int cursor;
    private int selectedSlot = -1;

    /// <summary>Number of preview cells refreshed per Tick (throttling knob).</summary>
    public int PreviewsPerFrame { get; set; } = 2;

    /// <summary>Preview RT aspect (width / height), so the camera feed is not distorted.</summary>
    public ShaderPreviewRenderer(int resolutionWidth, float aspect = 4f / 3f)
    {
        width = Mathf.Clamp(resolutionWidth, 64, 1024);
        height = ComputeHeight(width, aspect);

        // A 1-element zero buffer so shaders that declare StructuredBuffer<Detection>
        // have something bound (unbound structured buffers are undefined on some GPUs).
        int stride = Marshal.SizeOf(typeof(ShaderDetection));
        dummyDetectionBuffer = new ComputeBuffer(1, stride);
        dummyDetectionBuffer.SetData(new ShaderDetection[1]);
    }

    private static int ComputeHeight(int w, float aspect) =>
        Mathf.Clamp(Mathf.RoundToInt(w / Mathf.Max(0.1f, aspect)), 48, 1024);

    /// <summary>
    /// Updates preview texture aspect ratio when the live camera feed changes resolution
    /// (e.g. 1280x1280 1:1 on Horizon OS v83+ after a 4:3 fallback before camera start).
    /// </summary>
    public bool UpdateAspect(float aspect)
    {
        int newHeight = ComputeHeight(width, aspect);
        if (newHeight == height) return false;

        height = newHeight;
        for (int i = 0; i < rts.Count; i++)
        {
            if (rts[i] != null)
            {
                DestroyRenderTexture(rts[i]);
                rts[i] = CreatePreviewRT();
            }
        }
        return true;
    }

    /// <summary>
    /// Ensures a preview slot exists for the given shader material and returns its RT.
    /// Reuses existing RenderTextures and Material copies across menu refreshes when unchanged.
    /// </summary>
    public RenderTexture EnsureSlot(int slot, Material sourceMaterial)
    {
        if (slot < 0) return null;

        while (rts.Count <= slot)
        {
            rts.Add(null);
            mats.Add(null);
            sourceMats.Add(null);
        }

        if (rts[slot] == null || !rts[slot].IsCreated() || rts[slot].height != height)
        {
            if (rts[slot] != null) DestroyRenderTexture(rts[slot]);
            rts[slot] = CreatePreviewRT();
        }

        if (sourceMaterial != null && (mats[slot] == null || sourceMats[slot] != sourceMaterial))
        {
            if (mats[slot] != null) UnityEngine.Object.Destroy(mats[slot]);

            // Private copy so binding preview state never disturbs the shared history
            // material used by the main view.
            var m = new Material(sourceMaterial);
            m.SetFloat(EyeIndexProp, -1f); // render in 2D preview regardless of stereo eye gate
            BindZeroDetections(m);
            mats[slot] = m;
            sourceMats[slot] = sourceMaterial;
        }

        return rts[slot];
    }

    /// <summary>
    /// Trims excess preview slots when history shrinks without destroying reusable slots.
    /// </summary>
    public void TrimSlots(int count)
    {
        count = Mathf.Max(0, count);
        for (int i = rts.Count - 1; i >= count; i--)
        {
            if (rts[i] != null) DestroyRenderTexture(rts[i]);
            if (mats[i] != null) UnityEngine.Object.Destroy(mats[i]);
            rts.RemoveAt(i);
            mats.RemoveAt(i);
            sourceMats.RemoveAt(i);
        }

        if (cursor >= mats.Count) cursor = 0;
        if (selectedSlot >= mats.Count) selectedSlot = -1;
    }

    private RenderTexture CreatePreviewRT()
    {
        var rt = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
        {
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
            useMipMap = false,
        };
        rt.Create();
        return rt;
    }

    /// <summary>
    /// Marks one slot as "selected": only this slot is rendered every frame with real
    /// detections (via the bind callback in <see cref="Tick"/>); the previously selected
    /// slot reverts to a zero detection buffer (passthrough for detection shaders).
    /// </summary>
    public void SetSelectedSlot(int slot)
    {
        if (slot == selectedSlot) return;

        if (selectedSlot >= 0 && selectedSlot < mats.Count && mats[selectedSlot] != null)
        {
            BindZeroDetections(mats[selectedSlot]);
        }
        selectedSlot = slot;
        if (selectedSlot >= 0 && selectedSlot < mats.Count && mats[selectedSlot] != null)
        {
            BindZeroDetections(mats[selectedSlot]);
        }
    }

    /// <summary>
    /// Refreshes the selected slot every frame (with detections bound via
    /// <paramref name="bindSelectedDetections"/>) and round-robins a few of the rest with
    /// zero detections. Call once per frame while the menu is visible.
    /// </summary>
    public void Tick(Texture cameraFeed, System.Action<Material> bindSelectedDetections = null)
    {
        if (cameraFeed == null || mats.Count == 0) return;

        // Selected slot: always live, with real detections.
        if (selectedSlot >= 0 && selectedSlot < mats.Count
            && mats[selectedSlot] != null && rts[selectedSlot] != null)
        {
            var sm = mats[selectedSlot];
            bindSelectedDetections?.Invoke(sm);
            if (sm.HasProperty(BaseMapProp)) sm.SetTexture(BaseMapProp, cameraFeed);
            Graphics.Blit(cameraFeed, rts[selectedSlot], sm);
        }

        // Remaining slots: throttled round-robin, zero detections.
        int budget = Mathf.Min(Mathf.Max(1, PreviewsPerFrame), mats.Count);
        for (int k = 0; k < budget; k++)
        {
            cursor = (cursor + 1) % mats.Count;
            if (cursor == selectedSlot) continue;

            var m = mats[cursor];
            var rt = rts[cursor];
            if (m == null || rt == null) continue;

            if (m.HasProperty(BaseMapProp)) m.SetTexture(BaseMapProp, cameraFeed);
            Graphics.Blit(cameraFeed, rt, m);
        }
    }

    private void BindZeroDetections(Material m)
    {
        if (m == null) return;
        m.SetBuffer(DetectionsProp, dummyDetectionBuffer);
        m.SetInt(DetectionCountProp, 0);
        m.SetInt(HasMaskProp, 0);
        m.SetInt(MaskCountProp, 0);
    }

    private static void DestroyRenderTexture(RenderTexture rt)
    {
        if (rt == null) return;
        rt.Release();
        UnityEngine.Object.Destroy(rt);
    }

    /// <summary>
    /// Releases all slot textures and material copies, keeping the renderer reusable.
    /// </summary>
    public void ClearSlots()
    {
        foreach (var rt in rts)
        {
            if (rt != null) DestroyRenderTexture(rt);
        }
        foreach (var m in mats)
        {
            if (m != null) UnityEngine.Object.Destroy(m);
        }
        rts.Clear();
        mats.Clear();
        sourceMats.Clear();
        cursor = 0;
        selectedSlot = -1;
    }

    public void Dispose()
    {
        ClearSlots();
        dummyDetectionBuffer?.Release();
        dummyDetectionBuffer = null;
    }
}
