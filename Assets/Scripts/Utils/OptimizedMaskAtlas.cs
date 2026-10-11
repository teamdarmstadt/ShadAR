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
    private bool[] dirtySlots;
    private Texture[] slotTextures;
    private int[] slotMaskHashes;  // Hash of mask data to detect changes
    private int usedSlots;
    private bool atlasNeedsRebuild;
    
    // Pre-allocated temporary texture for blitting
    private RenderTexture tempRT;
    
    // Material for efficient blitting (with scale/offset support)
    private Material blitMaterial;
    
    // Command buffer for Quest-safe rendering
    private CommandBuffer commandBuffer;
    
    // Static shader property IDs for performance
    private static readonly int MainTexProp = Shader.PropertyToID("_MainTex");
    private static readonly int ColorProp = Shader.PropertyToID("_Color");
    
    // Black texture for clearing
    private Texture2D blackTexture;
    
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
    /// Gets whether the atlas has been modified since last frame.
    /// </summary>
    public bool WasModified { get; private set; }
    
    public OptimizedMaskAtlas(int maskSize = DefaultMaskSize)
    {
        currentMaskSize = maskSize;
        dirtySlots = new bool[MaxMasks];
        slotTextures = new Texture[MaxMasks];
        slotMaskHashes = new int[MaxMasks];
        
        // Create blit material - use a simple unlit shader
        var shader = Shader.Find("Unlit/Texture");
        if (shader == null) shader = Shader.Find("UI/Default");
        blitMaterial = new Material(shader);
        
        // Create command buffer for Quest-safe rendering
        commandBuffer = new CommandBuffer();
        commandBuffer.name = "MaskAtlas Update";
        
        // Create black texture for clearing slots
        blackTexture = new Texture2D(1, 1, TextureFormat.R8, false);
        blackTexture.SetPixel(0, 0, Color.black);
        blackTexture.Apply();
        
        // Initialize empty atlas
        RebuildAtlas(0);
    }
    
    /// <summary>
    /// Updates the atlas with new mask data.
    /// Only updates slots that have changed, avoiding full rebuild.
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
                WasModified = true;
            }
            return Array.Empty<int>();
        }
        
        int newMaskCount = Mathf.Min(masks.Count, MaxMasks);
        
        // Check if we need to resize atlas
        int requiredGridSize = Mathf.CeilToInt(Mathf.Sqrt(newMaskCount));
        if (requiredGridSize != currentGridSize || atlasNeedsRebuild)
        {
            RebuildAtlas(newMaskCount);
            WasModified = true;
        }
        
        int[] maskIndices = new int[newMaskCount];
        
        // Update each slot
        for (int i = 0; i < newMaskCount; i++)
        {
            Texture mask = masks[i];
            maskIndices[i] = i;
            
            // Calculate hash of texture to detect changes
            int newHash = ComputeTextureHash(mask);
            
            // Check if this slot needs update
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
        
        return maskIndices;
    }
    
    /// <summary>
    /// Computes a simple hash of texture properties for change detection.
    /// </summary>
    private int ComputeTextureHash(Texture tex)
    {
        if (tex == null) return 0;
        
        // Use instance ID + dimensions as a quick hash
        // This won't detect content changes but catches texture swaps
        return HashCode.Combine(
            tex.GetInstanceID(),
            tex.width,
            tex.height
        );
    }
    
    /// <summary>
    /// Rebuilds the atlas texture with new size.
    /// </summary>
    private void RebuildAtlas(int maskCount)
    {
        // Release old atlas
        if (atlasTexture != null)
        {
            atlasTexture.Release();
        }
        
        if (maskCount == 0)
        {
            // Create minimal 1x1 atlas
            currentGridSize = 1;
            currentAtlasSize = currentMaskSize;
        }
        else
        {
            currentGridSize = Mathf.CeilToInt(Mathf.Sqrt(maskCount));
            currentAtlasSize = currentGridSize * currentMaskSize;
        }
        
        // Create atlas with R8 format (single channel, 1 byte per pixel)
        atlasTexture = new RenderTexture(currentAtlasSize, currentAtlasSize, 0, RenderTextureFormat.R8);
        atlasTexture.filterMode = FilterMode.Bilinear;
        atlasTexture.wrapMode = TextureWrapMode.Clamp;
        atlasTexture.Create();
        
        // Clear to black
        ClearAtlas();
        
        // Mark all slots dirty
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
    
    /// <summary>
    /// Updates a single slot in the atlas with new mask data.
    /// Uses CommandBuffer Blit for efficient, Quest-safe GPU copy.
    /// </summary>
    private void UpdateSlot(int slotIndex, Texture sourceMask)
    {
        if (sourceMask == null) 
        {
            ClearSlot(slotIndex);
            return;
        }
        
        // Safety check for atlas
        if (atlasTexture == null || !atlasTexture.IsCreated())
        {
            Debug.LogWarning("[OptimizedMaskAtlas] Atlas texture not ready, skipping slot update");
            return;
        }
        
        int gridX = slotIndex % currentGridSize;
        int gridY = slotIndex / currentGridSize;
        
        // Calculate pixel coordinates for the slot
        int pixelX = gridX * currentMaskSize;
        int pixelY = gridY * currentMaskSize;
        
        // Create or resize temp RT if needed
        if (tempRT == null || tempRT.width != currentMaskSize || tempRT.height != currentMaskSize)
        {
            if (tempRT != null) tempRT.Release();
            tempRT = new RenderTexture(currentMaskSize, currentMaskSize, 0, RenderTextureFormat.R8);
            tempRT.enableRandomWrite = false;  // Not needed for blit target
            tempRT.Create();
        }
        
        // First blit source to temp (handles any size/format conversion)
        commandBuffer.Clear();
        commandBuffer.Blit(sourceMask, tempRT);
        Graphics.ExecuteCommandBuffer(commandBuffer);
        
        // Try CopyTexture first (most efficient), fall back to Blit if it fails
        try
        {
            if (SystemInfo.copyTextureSupport != CopyTextureSupport.None && 
                tempRT.IsCreated())
            {
                // Use CopyTexture for efficient GPU copy to specific region
                Graphics.CopyTexture(
                    tempRT, 0, 0, 0, 0, currentMaskSize, currentMaskSize,  // src
                    atlasTexture, 0, 0, pixelX, pixelY                      // dst
                );
            }
            else
            {
                // Fallback: Use Blit with a viewport (less efficient but more compatible)
                BlitToRegion(tempRT, pixelX, pixelY);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[OptimizedMaskAtlas] CopyTexture failed, using fallback: {e.Message}");
            // Fallback on exception
            BlitToRegion(tempRT, pixelX, pixelY);
        }
    }
    
    /// <summary>
    /// Fallback method to blit a texture to a specific region using CommandBuffer.
    /// </summary>
    private void BlitToRegion(Texture source, int destX, int destY)
    {
        // Use a temporary full-size blit approach with proper offset
        // This is less efficient but more compatible
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = atlasTexture;
        
        // Create MaterialPropertyBlock to avoid creating new materials
        var scale = new Vector2((float)currentMaskSize / currentAtlasSize, (float)currentMaskSize / currentAtlasSize);
        var offset = new Vector2((float)destX / currentAtlasSize, (float)destY / currentAtlasSize);
        
        blitMaterial.mainTexture = source;
        
        // Use CommandBuffer with proper viewport
        commandBuffer.Clear();
        commandBuffer.SetRenderTarget(atlasTexture);
        commandBuffer.SetViewport(new Rect(destX, destY, currentMaskSize, currentMaskSize));
        commandBuffer.Blit(source, BuiltinRenderTextureType.CurrentActive, blitMaterial);
        Graphics.ExecuteCommandBuffer(commandBuffer);
        
        RenderTexture.active = prev;
    }
    
    /// <summary>
    /// Clears a single slot to black.
    /// Uses CommandBuffer for Quest/Android compatibility.
    /// </summary>
    private void ClearSlot(int slotIndex)
    {
        // Safety check for atlas
        if (atlasTexture == null || !atlasTexture.IsCreated())
        {
            return;
        }
        
        int gridX = slotIndex % currentGridSize;
        int gridY = slotIndex / currentGridSize;
        int pixelX = gridX * currentMaskSize;
        int pixelY = gridY * currentMaskSize;
        
        // Create or resize temp RT if needed for clearing
        if (tempRT == null || tempRT.width != currentMaskSize || tempRT.height != currentMaskSize)
        {
            if (tempRT != null) tempRT.Release();
            tempRT = new RenderTexture(currentMaskSize, currentMaskSize, 0, RenderTextureFormat.R8);
            tempRT.Create();
        }
        
        // Clear temp RT to black
        commandBuffer.Clear();
        commandBuffer.SetRenderTarget(tempRT);
        commandBuffer.ClearRenderTarget(true, true, Color.black);
        Graphics.ExecuteCommandBuffer(commandBuffer);
        
        // Try CopyTexture first, fall back if needed
        try
        {
            if (SystemInfo.copyTextureSupport != CopyTextureSupport.None &&
                tempRT.IsCreated())
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
        catch (System.Exception)
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
        if (resolution != currentMaskSize)
        {
            currentMaskSize = resolution;
            atlasNeedsRebuild = true;
        }
    }
    
    /// <summary>
    /// Releases all resources.
    /// </summary>
    public void Dispose()
    {
        if (atlasTexture != null)
        {
            atlasTexture.Release();
            atlasTexture = null;
        }
        
        if (tempRT != null)
        {
            tempRT.Release();
            tempRT = null;
        }
        
        if (blitMaterial != null)
        {
            UnityEngine.Object.Destroy(blitMaterial);
            blitMaterial = null;
        }
        
        if (commandBuffer != null)
        {
            commandBuffer.Dispose();
            commandBuffer = null;
        }
        
        if (blackTexture != null)
        {
            UnityEngine.Object.Destroy(blackTexture);
            blackTexture = null;
        }
    }
}
