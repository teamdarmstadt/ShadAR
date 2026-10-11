using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// High-performance GPU-accelerated polygon mask rasterizer.
/// Uses compute shader for parallel rasterization instead of CPU scanline fill.
/// Falls back to optimized CPU implementation when GPU is unavailable.
/// </summary>
public class GPUPolygonRasterizer : IDisposable
{
    private ComputeShader computeShader;
    private int rasterizeKernel;
    private int clearKernel;
    
    // Pre-allocated buffers to avoid per-frame allocations
    private ComputeBuffer polygonBuffer;
    private float[] polygonDataArray;
    private int maxPolygonPoints;
    
    // CPU fallback buffers (pre-allocated)
    private byte[] cpuPixelBuffer;
    private int cpuBufferResolution;
    private List<float> cpuIntersections;
    
    // Texture pool for mask textures
    private Queue<RenderTexture> rtPool;
    private int pooledResolution;
    private const int MaxPooledTextures = 32;
    
    private bool useGPU;
    private bool isInitialized;
    
    /// <summary>
    /// Initializes the GPU polygon rasterizer.
    /// </summary>
    /// <param name="shader">PolygonRasterizer compute shader</param>
    /// <param name="maxPoints">Maximum polygon points to support (default 256)</param>
    public GPUPolygonRasterizer(ComputeShader shader = null, int maxPoints = 256)
    {
        maxPolygonPoints = maxPoints;
        rtPool = new Queue<RenderTexture>();
        cpuIntersections = new List<float>(64);
        
        if (shader != null && SystemInfo.supportsComputeShaders)
        {
            computeShader = shader;
            rasterizeKernel = computeShader.FindKernel("RasterizePolygon");
            clearKernel = computeShader.FindKernel("ClearTexture");
            
            // Pre-allocate polygon data buffer
            // Format: [x0, y0, x1, y1, ...] = maxPoints * 2 floats
            polygonDataArray = new float[maxPoints * 2];
            polygonBuffer = new ComputeBuffer(maxPoints * 2, sizeof(float));
            
            useGPU = true;
            Debug.Log("[GPUPolygonRasterizer] Initialized with GPU compute shader");
        }
        else
        {
            useGPU = false;
            Debug.Log("[GPUPolygonRasterizer] GPU compute not available, using optimized CPU fallback");
        }
        
        isInitialized = true;
    }
    
    /// <summary>
    /// Rasterizes a polygon mask to a RenderTexture (GPU) or Texture2D (CPU fallback).
    /// </summary>
    /// <param name="polygon">Polygon vertices in normalized [0,1] coordinates</param>
    /// <param name="resolution">Output texture resolution</param>
    /// <returns>Rasterized mask texture</returns>
    public Texture RasterizePolygon(Vector2[] polygon, int resolution)
    {
        if (!isInitialized || polygon == null || polygon.Length < 3)
            return null;
        
        if (useGPU)
        {
            return RasterizeGPU(polygon, resolution);
        }
        else
        {
            return RasterizeCPU(polygon, resolution);
        }
    }
    
    /// <summary>
    /// Rasterizes polygon using GPU compute shader.
    /// </summary>
    private RenderTexture RasterizeGPU(Vector2[] polygon, int resolution)
    {
        // Get or create RenderTexture
        RenderTexture rt = GetPooledRenderTexture(resolution);
        
        // Copy polygon data to pre-allocated array
        int numPoints = Mathf.Min(polygon.Length, maxPolygonPoints);
        for (int i = 0; i < numPoints; i++)
        {
            polygonDataArray[i * 2] = polygon[i].x;
            polygonDataArray[i * 2 + 1] = polygon[i].y;
        }
        polygonBuffer.SetData(polygonDataArray, 0, 0, numPoints * 2);
        
        // Clear texture first
        computeShader.SetTexture(clearKernel, "Result", rt);
        computeShader.SetInt("TextureSize", resolution);
        int clearGroups = Mathf.CeilToInt(resolution / 8f);
        computeShader.Dispatch(clearKernel, clearGroups, clearGroups, 1);
        
        // Rasterize polygon
        computeShader.SetTexture(rasterizeKernel, "Result", rt);
        computeShader.SetBuffer(rasterizeKernel, "PolygonData", polygonBuffer);
        computeShader.SetInt("TextureSize", resolution);
        computeShader.SetInt("PolygonOffset", 0);
        computeShader.SetInt("NumPoints", numPoints);
        
        int groups = Mathf.CeilToInt(resolution / 8f);
        computeShader.Dispatch(rasterizeKernel, groups, groups, 1);
        
        return rt;
    }
    
    /// <summary>
    /// Rasterizes polygon using optimized CPU scanline fill.
    /// Pre-allocates buffers to avoid GC pressure.
    /// </summary>
    private Texture2D RasterizeCPU(Vector2[] polygon, int resolution)
    {
        // Ensure CPU buffer is allocated and correct size
        if (cpuPixelBuffer == null || cpuBufferResolution != resolution)
        {
            cpuPixelBuffer = new byte[resolution * resolution];
            cpuBufferResolution = resolution;
        }
        else
        {
            // Clear buffer (faster than Array.Clear for byte[])
            Array.Clear(cpuPixelBuffer, 0, cpuPixelBuffer.Length);
        }
        
        // Convert normalized coordinates to pixel coordinates
        Vector2[] pixelPolygon = new Vector2[polygon.Length];
        for (int i = 0; i < polygon.Length; i++)
        {
            pixelPolygon[i] = new Vector2(
                polygon[i].x * (resolution - 1),
                (1f - polygon[i].y) * (resolution - 1)
            );
        }
        
        // Scanline fill algorithm with pre-allocated intersection list
        for (int y = 0; y < resolution; y++)
        {
            cpuIntersections.Clear();
            
            for (int i = 0; i < pixelPolygon.Length; i++)
            {
                int j = (i + 1) % pixelPolygon.Length;
                Vector2 p1 = pixelPolygon[i];
                Vector2 p2 = pixelPolygon[j];
                
                if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y))
                {
                    float t = (y - p1.y) / (p2.y - p1.y);
                    float x = p1.x + t * (p2.x - p1.x);
                    cpuIntersections.Add(x);
                }
            }
            
            cpuIntersections.Sort();
            
            for (int i = 0; i < cpuIntersections.Count - 1; i += 2)
            {
                int x1 = Mathf.Max(0, Mathf.CeilToInt(cpuIntersections[i]));
                int x2 = Mathf.Min(resolution - 1, Mathf.FloorToInt(cpuIntersections[i + 1]));
                
                int rowOffset = y * resolution;
                for (int x = x1; x <= x2; x++)
                {
                    cpuPixelBuffer[rowOffset + x] = 255;
                }
            }
        }
        
        // Create texture (consider pooling these too)
        var texture = new Texture2D(resolution, resolution, TextureFormat.R8, false);
        texture.filterMode = FilterMode.Bilinear;
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.LoadRawTextureData(cpuPixelBuffer);
        texture.Apply(false, false); // Don't generate mipmaps, keep readable
        
        return texture;
    }
    
    /// <summary>
    /// Gets a RenderTexture from the pool or creates a new one.
    /// </summary>
    private RenderTexture GetPooledRenderTexture(int resolution)
    {
        // Check if pool has correct resolution
        if (pooledResolution != resolution)
        {
            // Resolution changed, clear pool
            while (rtPool.Count > 0)
            {
                var old = rtPool.Dequeue();
                if (old != null) old.Release();
            }
            pooledResolution = resolution;
        }
        
        // Try to get from pool
        while (rtPool.Count > 0)
        {
            var pooled = rtPool.Dequeue();
            if (pooled != null && pooled.IsCreated())
            {
                return pooled;
            }
        }
        
        // Create new RenderTexture with R8 format for single-channel mask
        var rt = new RenderTexture(resolution, resolution, 0, RenderTextureFormat.R8);
        rt.enableRandomWrite = true;
        rt.filterMode = FilterMode.Bilinear;
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.Create();
        
        return rt;
    }
    
    /// <summary>
    /// Returns a RenderTexture to the pool for reuse.
    /// </summary>
    public void ReturnToPool(RenderTexture rt)
    {
        if (rt == null || !rt.IsCreated()) return;
        
        if (rt.width == pooledResolution && rtPool.Count < MaxPooledTextures)
        {
            rtPool.Enqueue(rt);
        }
        else
        {
            rt.Release();
        }
    }
    
    /// <summary>
    /// Releases all resources.
    /// </summary>
    public void Dispose()
    {
        polygonBuffer?.Release();
        polygonBuffer = null;
        
        while (rtPool.Count > 0)
        {
            var rt = rtPool.Dequeue();
            if (rt != null) rt.Release();
        }
        
        polygonDataArray = null;
        cpuPixelBuffer = null;
        isInitialized = false;
    }
}
