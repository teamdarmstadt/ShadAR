using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// High-performance mask atlas manager with dirty-flag system.
/// Only updates atlas regions that have changed, avoiding full atlas rebuild every frame.
/// Uses R8 texture format throughout for 4x less memory than RGBA32.
/// Uses CommandBuffer for Quest/Android compatibility (no GL.* calls).
/// </summary>
public class OptimizedMaskAtlas : IDisposable
{
    // Atlas configuration
    private const int MaxMasks = 16;           // 4x4 grid maximum
    private const int DefaultMaskSize = 256;   // Lower default for better performance

    // Atlas texture (R8 format for single-channel masks)
    private RenderTexture atlasTexture;
    private int currentAtlasSize;
    private int currentGridSize;
    private int currentMaskSize;

    // Dirty tracking
    private readonly bool[] dirtySlots;
    private readonly Texture[] slotTextures;
    private readonly int[] slotMaskHashes;  // Hash of mask identity + updateCount to detect changes
    private readonly int[][] cachedIndexArrays; // Pre-allocated [0..k-1] arrays to avoid per-call GC
    private int usedSlots;
    private bool atlasNeedsRebuild;

    // Pre-allocated temporary texture for blitting
    private RenderTexture tempRT;

    // Material for fallback blitting
    private Material blitMaterial;

    // Command buffer for Quest-safe rendering
    private CommandBuffer commandBuffer;

    /// <summary>
    /// Gets the atlas texture for shader binding.
    /// Returns null if atlas is not properly initialized.
    /// </summary>
    public Texture AtlasTexture
    {
        get
        {
            if (atlasTexture != null && atlasTexture.IsCreated())
                return atlasTexture;
            return null;
        }
    }

    /// <summary>
    /// Gets the number of masks currently in the atlas.
    /// </summary>
    public int MaskCount => usedSlots;

    /// <summary>
    /// Gets the grid dimension (e.g., 4 for 4x4 grid).
    /// </summary>
    public int GridSize => currentGridSize;

    /// <summary>
    /// Gets whether the atlas has been modified since last update.
    /// </summary>
    public bool WasModified { get; private set; }

    public OptimizedMaskAtlas(int maskSize = DefaultMaskSize)
    {
        currentMaskSize = Mathf.Max(16, maskSize);
        dirtySlots = new bool[MaxMasks];
        slotTextures = new Texture[MaxMasks];
        slotMaskHashes = new int[MaxMasks];

        cachedIndexArrays = new int[MaxMasks + 1][];
        cachedIndexArrays[0] = Array.Empty<int>();
        for (int k = 1; k <= MaxMasks; k++)
        {
            var indices = new int[k];
            for (int i = 0; i < k; i++) indices[i] = i;
            cachedIndexArrays[k] = indices;
        }

        // Create blit material - use a simple unlit shader
        var shader = Shader.Find("Unlit/Texture");
        if (shader == null) shader = Shader.Find("UI/Default");
        blitMaterial = new Material(shader);

        // Create command buffer for Quest-safe rendering
        commandBuffer = new CommandBuffer { name = "MaskAtlas Update" };

        // Initialize empty atlas
        RebuildAtlas(0);
    }

    /// <summary>
    /// Updates the atlas with new mask data.
    /// Only updates slots that have changed, avoiding full rebuild and heap allocations.
    /// </summary>
    /// <param name="masks">List of mask textures (can be Texture2D or RenderTexture)</param>
    /// <returns>Array of mask indices for each input mask (for shader binding)</returns>
    public int[] UpdateMasks(List<Texture> masks)
    {
        WasModified = false;

        if (masks == null || masks.Count == 0)
        {
            if (usedSlots > 0)
            {
                ClearAtlas();
                for (int i = 0; i < MaxMasks; i++)
                {
                    slotTextures[i] = null;
                    slotMaskHashes[i] = 0;
                }
                WasModified = true;
            }
            return cachedIndexArrays[0];
        }

        int newMaskCount = Mathf.Min(masks.Count, MaxMasks);

        // Check if we need to resize atlas
        int requiredGridSize = Mathf.CeilToInt(Mathf.Sqrt(newMaskCount));
        if (requiredGridSize != currentGridSize || atlasNeedsRebuild || atlasTexture == null || !atlasTexture.IsCreated())
        {
            RebuildAtlas(newMaskCount);
            WasModified = true;
        }

        // Update each slot whose texture reference or content updateCount changed
        for (int i = 0; i < newMaskCount; i++)
        {
            Texture mask = masks[i];
            int newHash = ComputeTextureHash(mask);

            bool needsUpdate = dirtySlots[i] ||
                               slotTextures[i] != mask ||
                               slotMaskHashes[i] != newHash;

            if (needsUpdate)
            {
                UpdateSlot(i, mask);
                slotTextures[i] = mask;
                slotMaskHashes[i] = newHash;
                dirtySlots[i] = false;
                WasModified = true;
            }
        }

        // Clear unused slots
        for (int i = newMaskCount; i < usedSlots; i++)
        {
            ClearSlot(i);
            slotTextures[i] = null;
            slotMaskHashes[i] = 0;
            WasModified = true;
        }

        usedSlots = newMaskCount;

        return cachedIndexArrays[newMaskCount];
    }

    /// <summary>
    /// Computes a hash of texture identity + updateCount so pooled textures whose contents were
    /// re-rendered (via IncrementUpdateCount / Apply) trigger an atlas slot refresh.
    /// </summary>
    private static int ComputeTextureHash(Texture tex)
    {
        if (tex == null) return 0;

        return HashCode.Combine(
            tex.GetHashCode(),
            (int)tex.updateCount,
            tex.width,
            tex.height
        );
    }

    /// <summary>
    /// Rebuilds the atlas texture with new size.
    /// </summary>
    private void RebuildAtlas(int maskCount)
    {
        DestroyRenderTexture(ref atlasTexture);

        if (maskCount == 0)
        {
            currentGridSize = 1;
            currentAtlasSize = currentMaskSize;
        }
        else
        {
            currentGridSize = Mathf.CeilToInt(Mathf.Sqrt(maskCount));
            currentAtlasSize = currentGridSize * currentMaskSize;
        }

        // Create atlas with R8 format (single channel, 1 byte per pixel)
        atlasTexture = new RenderTexture(currentAtlasSize, currentAtlasSize, 0, RenderTextureFormat.R8)
        {
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        atlasTexture.Create();

        ClearAtlas();

        for (int i = 0; i < MaxMasks; i++)
        {
            dirtySlots[i] = true;
        }

        atlasNeedsRebuild = false;
    }

    /// <summary>
    /// Clears the entire atlas to black.
    /// Uses CommandBuffer for Quest/Android compatibility.
    /// </summary>
    private void ClearAtlas()
    {
        commandBuffer.Clear();
        commandBuffer.SetRenderTarget(atlasTexture);
        commandBuffer.ClearRenderTarget(true, true, Color.black);
        Graphics.ExecuteCommandBuffer(commandBuffer);
        usedSlots = 0;
    }

    private void EnsureTempRT()
    {
        if (tempRT != null && tempRT.IsCreated() && tempRT.width == currentMaskSize && tempRT.height == currentMaskSize)
            return;

        DestroyRenderTexture(ref tempRT);
        tempRT = new RenderTexture(currentMaskSize, currentMaskSize, 0, RenderTextureFormat.R8)
        {
            enableRandomWrite = false,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        tempRT.Create();
    }

    /// <summary>
    /// Updates a single slot in the atlas with new mask data.
    /// </summary>
    private void UpdateSlot(int slotIndex, Texture sourceMask)
    {
        if (sourceMask == null)
        {
            ClearSlot(slotIndex);
            return;
        }

        if (atlasTexture == null || !atlasTexture.IsCreated())
        {
            Debug.LogWarning("[OptimizedMaskAtlas] Atlas texture not ready, skipping slot update");
            return;
        }

        int gridX = slotIndex % currentGridSize;
        int gridY = slotIndex / currentGridSize;
        int pixelX = gridX * currentMaskSize;
        int pixelY = gridY * currentMaskSize;

        // Fast path: if sourceMask already matches cell size and R8 format, copy directly without tempRT blit.
        bool canDirectCopy = SystemInfo.copyTextureSupport != CopyTextureSupport.None
                             && sourceMask.width == currentMaskSize
                             && sourceMask.height == currentMaskSize
                             && ((sourceMask is RenderTexture srcRt && srcRt.format == RenderTextureFormat.R8 && srcRt.IsCreated())
                                 || (sourceMask is Texture2D srcTex && srcTex.format == TextureFormat.R8));

        if (canDirectCopy)
        {
            try
            {
                Graphics.CopyTexture(
                    sourceMask, 0, 0, 0, 0, currentMaskSize, currentMaskSize,
                    atlasTexture, 0, 0, pixelX, pixelY
                );
                return;
            }
            catch (Exception)
            {
                // Fall through to tempRT resample + copy
            }
        }

        EnsureTempRT();

        commandBuffer.Clear();
        commandBuffer.Blit(sourceMask, tempRT);
        Graphics.ExecuteCommandBuffer(commandBuffer);

        try
        {
            if (SystemInfo.copyTextureSupport != CopyTextureSupport.None && tempRT.IsCreated())
            {
                Graphics.CopyTexture(
                    tempRT, 0, 0, 0, 0, currentMaskSize, currentMaskSize,
                    atlasTexture, 0, 0, pixelX, pixelY
                );
            }
            else
            {
                BlitToRegion(tempRT, pixelX, pixelY);
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[OptimizedMaskAtlas] CopyTexture failed, using fallback: {e.Message}");
            BlitToRegion(tempRT, pixelX, pixelY);
        }
    }

    /// <summary>
    /// Fallback method to blit a texture to a specific region using CommandBuffer viewport.
    /// </summary>
    private void BlitToRegion(Texture source, int destX, int destY)
    {
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = atlasTexture;

        blitMaterial.mainTexture = source;

        commandBuffer.Clear();
        commandBuffer.SetRenderTarget(atlasTexture);
        commandBuffer.SetViewport(new Rect(destX, destY, currentMaskSize, currentMaskSize));
        commandBuffer.Blit(source, BuiltinRenderTextureType.CurrentActive, blitMaterial);
        Graphics.ExecuteCommandBuffer(commandBuffer);

        RenderTexture.active = prev;
    }

    /// <summary>
    /// Clears a single slot to black.
    /// </summary>
    private void ClearSlot(int slotIndex)
    {
        if (atlasTexture == null || !atlasTexture.IsCreated())
        {
            return;
        }

        int gridX = slotIndex % currentGridSize;
        int gridY = slotIndex / currentGridSize;
        int pixelX = gridX * currentMaskSize;
        int pixelY = gridY * currentMaskSize;

        EnsureTempRT();

        commandBuffer.Clear();
        commandBuffer.SetRenderTarget(tempRT);
        commandBuffer.ClearRenderTarget(true, true, Color.black);
        Graphics.ExecuteCommandBuffer(commandBuffer);

        try
        {
            if (SystemInfo.copyTextureSupport != CopyTextureSupport.None && tempRT.IsCreated())
            {
                Graphics.CopyTexture(
                    tempRT, 0, 0, 0, 0, currentMaskSize, currentMaskSize,
                    atlasTexture, 0, 0, pixelX, pixelY
                );
            }
            else
            {
                BlitToRegion(tempRT, pixelX, pixelY);
            }
        }
        catch (Exception)
        {
            BlitToRegion(tempRT, pixelX, pixelY);
        }
    }

    /// <summary>
    /// Forces a full atlas rebuild on next update.
    /// </summary>
    public void Invalidate()
    {
        atlasNeedsRebuild = true;
        for (int i = 0; i < MaxMasks; i++)
        {
            dirtySlots[i] = true;
        }
    }

    /// <summary>
    /// Sets the mask resolution. Triggers atlas rebuild if changed.
    /// </summary>
    public void SetMaskResolution(int resolution)
    {
        resolution = Mathf.Max(16, resolution);
        if (resolution != currentMaskSize)
        {
            currentMaskSize = resolution;
            atlasNeedsRebuild = true;
        }
    }

    private static void DestroyRenderTexture(ref RenderTexture rt)
    {
        if (rt == null) return;
        rt.Release();
        if (Application.isPlaying) UnityEngine.Object.Destroy(rt);
        else UnityEngine.Object.DestroyImmediate(rt);
        rt = null;
    }

    /// <summary>
    /// Releases all resources.
    /// </summary>
    public void Dispose()
    {
        DestroyRenderTexture(ref atlasTexture);
        DestroyRenderTexture(ref tempRT);

        if (blitMaterial != null)
        {
            if (Application.isPlaying) UnityEngine.Object.Destroy(blitMaterial);
            else UnityEngine.Object.DestroyImmediate(blitMaterial);
            blitMaterial = null;
        }

        if (commandBuffer != null)
        {
            commandBuffer.Dispose();
            commandBuffer = null;
        }
    }
}
