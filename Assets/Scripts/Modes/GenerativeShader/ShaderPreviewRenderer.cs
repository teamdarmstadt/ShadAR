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
///
/// Note on detections: a zero-length detection buffer is bound (DetectionCount = 0), so
/// detection-driven shaders render plain passthrough in previews (their highlights only
/// exist for the one vocabulary currently active on the server). Full-frame effect
/// shaders preview correctly.
/// </summary>
public class ShaderPreviewRenderer : IDisposable
{
    private static readonly int MainTexProp = Shader.PropertyToID("_MainTex");
    private static readonly int BaseMapProp = Shader.PropertyToID("_BaseMap");
    private static readonly int DetectionsProp = Shader.PropertyToID("_Detections");
    private static readonly int DetectionCountProp = Shader.PropertyToID("_DetectionCount");
    private static readonly int HasMaskProp = Shader.PropertyToID("_HasMask");
    private static readonly int MaskCountProp = Shader.PropertyToID("_MaskCount");

    private readonly int width;
    private readonly int height;
    private readonly List<RenderTexture> rts = new();
    private readonly List<Material> mats = new();
    private ComputeBuffer dummyDetectionBuffer;
    private int cursor;
    private int selectedSlot = -1;

    /// <summary>Number of preview cells refreshed per Tick (throttling knob).</summary>
    public int PreviewsPerFrame { get; set; } = 2;

    /// <summary>Preview RT aspect (width / height), so the camera feed is not distorted.</summary>
    public ShaderPreviewRenderer(int resolutionWidth, float aspect = 4f / 3f)
    {
        width = Mathf.Clamp(resolutionWidth, 64, 1024);
        height = Mathf.Clamp(Mathf.RoundToInt(width / Mathf.Max(0.1f, aspect)), 48, 1024);

        // A 1-element zero buffer so shaders that declare StructuredBuffer<Detection>
        // have something bound (unbound structured buffers are undefined on some GPUs).
        int stride = Marshal.SizeOf(typeof(ShaderDetection));
        dummyDetectionBuffer = new ComputeBuffer(1, stride);
        dummyDetectionBuffer.SetData(new ShaderDetection[1]);
    }

    /// <summary>
    /// Ensures a preview slot exists for the given shader material and returns its RT.
    /// Safe to call repeatedly (idempotent per slot).
    /// </summary>
    public RenderTexture EnsureSlot(int slot, Material sourceMaterial)
    {
        if (slot < 0) return null;

        while (rts.Count <= slot)
        {
            rts.Add(null);
            mats.Add(null);
        }

        if (rts[slot] == null)
        {
            var rt = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                useMipMap = false,
            };
            rt.Create();
            rts[slot] = rt;
        }

        if (mats[slot] == null && sourceMaterial != null)
        {
            // Private copy so binding preview state never disturbs the shared history
            // material used by the main view.
            var m = new Material(sourceMaterial);
            BindZeroDetections(m);
            mats[slot] = m;
        }

        return rts[slot];
    }

    /// <summary>
    /// Marks one slot as "selected": only this slot is rendered every frame with real
    /// detections (via the bind callback in <see cref="Tick"/>); the previously selected
    /// slot reverts to a zero detection buffer (passthrough for detection shaders).
    /// </summary>
    public void SetSelectedSlot(int slot)
    {
        if (slot == selectedSlot) return;

        // Revert the previous selection, and start the new one clean (passthrough) until
        // its detections are (re)bound by the caller.
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
            try { sm.SetTexture(BaseMapProp, cameraFeed); } catch { }
            Graphics.Blit(cameraFeed, rts[selectedSlot], sm);
        }

        // Remaining slots: throttled round-robin, zero detections.
        int budget = Mathf.Max(1, PreviewsPerFrame);
        for (int k = 0; k < budget; k++)
        {
            cursor = (cursor + 1) % mats.Count;
            if (cursor == selectedSlot) continue;

            var m = mats[cursor];
            var rt = rts[cursor];
            if (m == null || rt == null) continue;

            try { m.SetTexture(BaseMapProp, cameraFeed); } catch { }
            Graphics.Blit(cameraFeed, rt, m);
        }
    }

    private void BindZeroDetections(Material m)
    {
        if (m == null) return;
        try { m.SetBuffer(DetectionsProp, dummyDetectionBuffer); } catch { /* no _Detections */ }
        try { m.SetInt(DetectionCountProp, 0); } catch { }
        try { m.SetInt(HasMaskProp, 0); } catch { }
        try { m.SetInt(MaskCountProp, 0); } catch { }
    }

    /// <summary>
    /// Releases all slot textures and material copies (e.g. when the history changes),
    /// keeping the renderer reusable.
    /// </summary>
    public void ClearSlots()
    {
        foreach (var rt in rts)
        {
            if (rt != null) rt.Release();
        }
        foreach (var m in mats)
        {
            if (m != null) UnityEngine.Object.Destroy(m);
        }
        rts.Clear();
        mats.Clear();
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
