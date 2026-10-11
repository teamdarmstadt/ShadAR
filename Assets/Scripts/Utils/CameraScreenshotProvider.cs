using System;
using System.Collections;
using UnityEngine;

/// <summary>
/// Grabs downscaled CPU copies of the passthrough feed for the detection stream. The feed is
/// registered by <see cref="ShaderOverlayLoader"/>; buffers are reused, so a capture allocates
/// nothing once the size is stable.
/// </summary>
public class CameraScreenshotProvider : MonoBehaviour
{
    [Tooltip("Width of the frames sent to the detector (0 = full resolution); height keeps the aspect.")]
    [SerializeField] private int streamingResolution = 512;

    public static CameraScreenshotProvider Instance { get; private set; }

    private Func<Texture> cameraTextureProvider;
    private RenderTexture streamingRT;
    private Texture2D streamingTexture;

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
        if (streamingRT != null) streamingRT.Release();
        if (streamingTexture != null) Destroy(streamingTexture);
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
    /// Captures a downscaled frame at the end of the current frame.
    /// </summary>
    /// <param name="onComplete">Receives the reused Texture2D (do not destroy it), or null when no
    /// feed is available yet.</param>
    public void CaptureStreamingScreenshot(Action<Texture2D> onComplete)
    {
        StartCoroutine(CaptureStreamingCoroutine(onComplete));
    }

    private IEnumerator CaptureStreamingCoroutine(Action<Texture2D> onComplete)
    {
        Texture sourceTexture = cameraTextureProvider?.Invoke();
        if (sourceTexture == null)
        {
            onComplete?.Invoke(null);
            yield break;
        }

        yield return new WaitForEndOfFrame();

        int targetWidth = streamingResolution > 0 ? streamingResolution : sourceTexture.width;
        int targetHeight = streamingResolution > 0
            ? (int)(sourceTexture.height * ((float)streamingResolution / sourceTexture.width))
            : sourceTexture.height;

        if (streamingRT == null || streamingRT.width != targetWidth || streamingRT.height != targetHeight)
        {
            if (streamingRT != null) streamingRT.Release();
            streamingRT = new RenderTexture(targetWidth, targetHeight, 0, RenderTextureFormat.Default);
            streamingRT.Create();
        }

        Graphics.Blit(sourceTexture, streamingRT);

        if (streamingTexture == null || streamingTexture.width != targetWidth || streamingTexture.height != targetHeight)
        {
            if (streamingTexture != null) Destroy(streamingTexture);
            streamingTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
        }

        // ponytail: synchronous GPU readback of a small frame; AsyncGPUReadback would remove the
        // stall if detection capture ever shows up in the frame time.
        RenderTexture.active = streamingRT;
        streamingTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        streamingTexture.Apply();
        RenderTexture.active = null;

        onComplete?.Invoke(streamingTexture);
    }
}
