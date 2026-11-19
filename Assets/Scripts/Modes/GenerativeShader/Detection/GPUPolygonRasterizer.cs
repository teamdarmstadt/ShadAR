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
    private static readonly int ResultProp = Shader.PropertyToID("Result");
    private static readonly int PolygonDataProp = Shader.PropertyToID("PolygonData");
    private static readonly int TextureSizeProp = Shader.PropertyToID("TextureSize");
    private static readonly int PolygonOffsetProp = Shader.PropertyToID("PolygonOffset");
    private static readonly int NumPointsProp = Shader.PropertyToID("NumPoints");

    private ComputeShader computeShader;
    private int rasterizeKernel;

    // Pre-allocated buffers to avoid per-frame allocations
    private ComputeBuffer polygonBuffer;
    private float[] polygonDataArray;
    private int maxPolygonPoints;

    // CPU fallback buffers (pre-allocated)
    private byte[] cpuPixelBuffer;
    private int cpuBufferResolution;
    private List<float> cpuIntersections;
    private Vector2[] cpuPixelPolygon;

    // Bounded texture pools (also auto-recycles oldest active textures if callers omit ReturnToPool)
    private readonly Queue<RenderTexture> rtPool = new();
    private readonly HashSet<RenderTexture> pooledRtSet = new();
    private readonly Queue<RenderTexture> activeRts = new();
    private readonly Queue<Texture2D> cpuTexPool = new();
    private readonly Queue<Texture2D> activeCpuTextures = new();
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
        maxPolygonPoints = Mathf.Max(3, maxPoints);
        cpuIntersections = new List<float>(64);
        cpuPixelPolygon = new Vector2[maxPolygonPoints];

        if (shader != null && SystemInfo.supportsComputeShaders)
        {
            computeShader = shader;
            rasterizeKernel = computeShader.FindKernel("RasterizePolygon");

            // Pre-allocate polygon data buffer: [x0, y0, x1, y1, ...] = maxPoints * 2 floats
            polygonDataArray = new float[maxPolygonPoints * 2];
            polygonBuffer = new ComputeBuffer(maxPolygonPoints * 2, sizeof(float));

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
        if (!isInitialized || polygon == null || polygon.Length < 3 || resolution <= 0)
            return null;

        return useGPU
            ? RasterizeGPU(polygon, resolution)
            : RasterizeCPU(polygon, resolution);
    }

    /// <summary>
    /// Rasterizes polygon using GPU compute shader.
    /// </summary>
    private RenderTexture RasterizeGPU(Vector2[] polygon, int resolution)
    {
        RenderTexture rt = GetPooledRenderTexture(resolution);

        int numPoints = Mathf.Min(polygon.Length, maxPolygonPoints);
        for (int i = 0; i < numPoints; i++)
        {
            polygonDataArray[i * 2] = polygon[i].x;
            polygonDataArray[i * 2 + 1] = polygon[i].y;
        }
        polygonBuffer.SetData(polygonDataArray, 0, 0, numPoints * 2);

        // RasterizePolygon writes every pixel (0.0 or 1.0), so no separate ClearTexture pass is needed.
        computeShader.SetTexture(rasterizeKernel, ResultProp, rt);
        computeShader.SetBuffer(rasterizeKernel, PolygonDataProp, polygonBuffer);
        computeShader.SetInt(TextureSizeProp, resolution);
        computeShader.SetInt(PolygonOffsetProp, 0);
        computeShader.SetInt(NumPointsProp, numPoints);

        int groups = Mathf.CeilToInt(resolution / 8f);
        computeShader.Dispatch(rasterizeKernel, groups, groups, 1);

        // Bump updateCount so OptimizedMaskAtlas detects content changes on pooled RenderTextures.
        rt.IncrementUpdateCount();
        return rt;
    }

    /// <summary>
    /// Rasterizes polygon using optimized CPU scanline fill.
    /// Pre-allocates buffers and pools textures to avoid GC and native memory leaks.
    /// </summary>
    private Texture2D RasterizeCPU(Vector2[] polygon, int resolution)
    {
        EnsurePoolResolution(resolution);

        if (cpuPixelBuffer == null || cpuBufferResolution != resolution)
        {
            cpuPixelBuffer = new byte[resolution * resolution];
            cpuBufferResolution = resolution;
        }
        else
        {
            Array.Clear(cpuPixelBuffer, 0, cpuPixelBuffer.Length);
        }

        int numPoints = polygon.Length;
        if (cpuPixelPolygon == null || cpuPixelPolygon.Length < numPoints)
        {
            cpuPixelPolygon = new Vector2[numPoints];
        }

        float scale = Mathf.Max(resolution - 1, 1);
        for (int i = 0; i < numPoints; i++)
        {
            cpuPixelPolygon[i] = new Vector2(
                polygon[i].x * scale,
                (1f - polygon[i].y) * scale
            );
        }

        for (int y = 0; y < resolution; y++)
        {
            cpuIntersections.Clear();

            for (int i = 0; i < numPoints; i++)
            {
                int j = (i + 1) % numPoints;
                Vector2 p1 = cpuPixelPolygon[i];
                Vector2 p2 = cpuPixelPolygon[j];

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

        Texture2D texture = GetPooledTexture2D(resolution);
        texture.LoadRawTextureData(cpuPixelBuffer);
        texture.Apply(false, false); // Also increments Texture.updateCount
        return texture;
    }

    private void EnsurePoolResolution(int resolution)
    {
        if (pooledResolution == resolution) return;
        ClearAllTextures();
        pooledResolution = resolution;
    }

    /// <summary>
    /// Gets a RenderTexture from the pool, recycles the oldest active texture if the pool cap is
    /// reached, or creates a new one.
    /// </summary>
    private RenderTexture GetPooledRenderTexture(int resolution)
    {
        EnsurePoolResolution(resolution);

        while (rtPool.Count > 0)
        {
            var pooled = rtPool.Dequeue();
            pooledRtSet.Remove(pooled);
            if (pooled != null && pooled.IsCreated())
            {
                activeRts.Enqueue(pooled);
                return pooled;
            }
            DestroyUnityObject(pooled);
        }

        // Guard against unbounded VRAM/native leaks if callers omit ReturnToPool: recycle the
        // oldest active RenderTexture once we have MaxPooledTextures in rotation.
        while (activeRts.Count >= MaxPooledTextures)
        {
            var oldest = activeRts.Dequeue();
            if (oldest == null || pooledRtSet.Contains(oldest)) continue;
            if (oldest.IsCreated() && oldest.width == resolution)
            {
                activeRts.Enqueue(oldest);
                return oldest;
            }
            DestroyUnityObject(oldest);
        }

        var rt = new RenderTexture(resolution, resolution, 0, RenderTextureFormat.R8)
        {
            enableRandomWrite = true,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        rt.Create();
        activeRts.Enqueue(rt);
        return rt;
    }

    private Texture2D GetPooledTexture2D(int resolution)
    {
        while (cpuTexPool.Count > 0)
        {
            var pooled = cpuTexPool.Dequeue();
            if (pooled != null && pooled.width == resolution)
            {
                activeCpuTextures.Enqueue(pooled);
                return pooled;
            }
            DestroyUnityObject(pooled);
        }

        while (activeCpuTextures.Count >= MaxPooledTextures)
        {
            var oldest = activeCpuTextures.Dequeue();
            if (oldest != null && oldest.width == resolution)
            {
                activeCpuTextures.Enqueue(oldest);
                return oldest;
            }
            DestroyUnityObject(oldest);
        }

        var tex = new Texture2D(resolution, resolution, TextureFormat.R8, false)
        {
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        activeCpuTextures.Enqueue(tex);
        return tex;
    }

    /// <summary>
    /// Returns a RenderTexture to the pool for reuse.
    /// </summary>
    public void ReturnToPool(RenderTexture rt)
    {
        if (rt == null) return;

        if (!rt.IsCreated() || rt.width != pooledResolution || rtPool.Count >= MaxPooledTextures)
        {
            pooledRtSet.Remove(rt);
            DestroyUnityObject(rt);
            return;
        }

        if (pooledRtSet.Add(rt))
        {
            rtPool.Enqueue(rt);
        }
    }

    private void ClearAllTextures()
    {
        var seen = new HashSet<UnityEngine.Object>();

        while (rtPool.Count > 0)
        {
            var rt = rtPool.Dequeue();
            if (rt != null && seen.Add(rt)) DestroyUnityObject(rt);
        }
        pooledRtSet.Clear();

        while (activeRts.Count > 0)
        {
            var rt = activeRts.Dequeue();
            if (rt != null && seen.Add(rt)) DestroyUnityObject(rt);
        }

        while (cpuTexPool.Count > 0)
        {
            var tex = cpuTexPool.Dequeue();
            if (tex != null && seen.Add(tex)) DestroyUnityObject(tex);
        }

        while (activeCpuTextures.Count > 0)
        {
            var tex = activeCpuTextures.Dequeue();
            if (tex != null && seen.Add(tex)) DestroyUnityObject(tex);
        }
    }

    private static void DestroyUnityObject(UnityEngine.Object obj)
    {
        if (obj == null) return;
        if (obj is RenderTexture rt) rt.Release();
        if (Application.isPlaying) UnityEngine.Object.Destroy(obj);
        else UnityEngine.Object.DestroyImmediate(obj);
    }

    /// <summary>
    /// Releases all resources.
    /// </summary>
    public void Dispose()
    {
        polygonBuffer?.Release();
        polygonBuffer = null;

        ClearAllTextures();

        polygonDataArray = null;
        cpuPixelBuffer = null;
        cpuPixelPolygon = null;
        isInitialized = false;
    }
}
