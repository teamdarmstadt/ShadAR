using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Grabs downscaled CPU copies of the passthrough feed for the detection stream. The feed is
/// registered by <see cref="ShaderOverlayLoader"/>; buffers are reused and GPU readback is
/// asynchronous via <see cref="AsyncGPUReadback"/> so frame capture does not stall VR rendering.
/// </summary>
public class CameraScreenshotProvider : MonoBehaviour
{
    [Tooltip("Width of the frames sent to the detector (0 = full resolution); height keeps the aspect.")]
    [SerializeField] private int streamingResolution = 512;

    public static CameraScreenshotProvider Instance { get; private set; }

    private Func<Texture> cameraTextureProvider;
    private RenderTexture streamingRT;
    private Texture2D streamingTexture;
    private bool captureInFlight;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(this);
            return;
        }
        Instance = this;
    }

    private void OnDestroy()
    {
        if (streamingRT != null)
        {
            streamingRT.Release();
            Destroy(streamingRT);
            streamingRT = null;
        }
        if (streamingTexture != null)
        {
            Destroy(streamingTexture);
            streamingTexture = null;
        }
        if (Instance == this) Instance = null;
    }

    /// <summary>
    /// Registers a function that returns the current camera texture (it can change over time,
    /// e.g. after the camera restarts).
    /// </summary>
    public void RegisterCameraScreenshotProvider(Func<Texture> provider)
    {
        cameraTextureProvider = provider;
    }

    /// <summary>
    /// Captures a downscaled frame without blocking the render thread when AsyncGPUReadback is supported.
    /// </summary>
    /// <param name="onComplete">Receives the reused Texture2D (do not destroy it), or null when no
    /// feed is available yet.</param>
    public void CaptureStreamingScreenshot(Action<Texture2D> onComplete)
    {
        if (captureInFlight)
        {
            onComplete?.Invoke(null);
            return;
        }

        Texture sourceTexture = cameraTextureProvider?.Invoke();
        if (sourceTexture == null || sourceTexture.width <= 0 || sourceTexture.height <= 0)
        {
            onComplete?.Invoke(null);
            return;
        }

        int targetWidth = streamingResolution > 0 ? streamingResolution : sourceTexture.width;
        int targetHeight = streamingResolution > 0
            ? Mathf.Max(1, Mathf.RoundToInt(sourceTexture.height * ((float)streamingResolution / sourceTexture.width)))
            : sourceTexture.height;

        EnsureStreamingBuffers(targetWidth, targetHeight);
        Graphics.Blit(sourceTexture, streamingRT);

        if (SystemInfo.supportsAsyncGPUReadback)
        {
            captureInFlight = true;
            AsyncGPUReadback.Request(streamingRT, 0, TextureFormat.RGB24, request =>
            {
                captureInFlight = false;
                if (this == null) return;

                if (request.hasError || streamingTexture == null
                    || streamingTexture.width != targetWidth || streamingTexture.height != targetHeight)
                {
                    onComplete?.Invoke(null);
                    return;
                }

                streamingTexture.LoadRawTextureData(request.GetData<byte>());
                onComplete?.Invoke(streamingTexture);
            });
        }
        else
        {
            StartCoroutine(CaptureStreamingFallbackCoroutine(targetWidth, targetHeight, onComplete));
        }
    }

    private void EnsureStreamingBuffers(int targetWidth, int targetHeight)
    {
        if (streamingRT == null || !streamingRT.IsCreated()
            || streamingRT.width != targetWidth || streamingRT.height != targetHeight)
        {
            if (streamingRT != null)
            {
                streamingRT.Release();
                Destroy(streamingRT);
            }
            streamingRT = new RenderTexture(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32);
            streamingRT.Create();
        }

        if (streamingTexture == null || streamingTexture.width != targetWidth || streamingTexture.height != targetHeight)
        {
            if (streamingTexture != null) Destroy(streamingTexture);
            streamingTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
        }
    }

    private IEnumerator CaptureStreamingFallbackCoroutine(int targetWidth, int targetHeight, Action<Texture2D> onComplete)
    {
        captureInFlight = true;
        yield return new WaitForEndOfFrame();
        captureInFlight = false;

        if (this == null || streamingRT == null || streamingTexture == null)
        {
            onComplete?.Invoke(null);
            yield break;
        }

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = streamingRT;
        streamingTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        streamingTexture.Apply(false, false);
        RenderTexture.active = prev;

        onComplete?.Invoke(streamingTexture);
    }
}
