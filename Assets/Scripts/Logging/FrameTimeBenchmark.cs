using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// Records camera-feed frame intervals for a specified duration after an event (e.g. shader applied).
/// Measures the time between consecutive camera-texture updates (not Unity render frames).
/// Computes statistics: avg, sd, min, max, p95, total camera frames.
/// Attach to a GameObject and reference from PipelineBenchmarkLogger.
/// </summary>
public class FrameTimeBenchmark : MonoBehaviour
{
    private bool isRecording;
    private readonly List<float> frameTimes = new();
    private float remainingTime;
    private Action<FrameTimeStats> onComplete;
    private Func<bool> cameraUpdatedCheck;
    private float lastCameraUpdateTime;
    private bool hasPreviousUpdate;

    /// <summary>
    /// Whether frame time recording is currently active.
    /// </summary>
    public bool IsRecording => isRecording;

    /// <summary>
    /// Starts recording camera-feed frame intervals for the given duration.
    /// <paramref name="cameraUpdated"/> should return true on frames where the camera texture was refreshed
    /// (e.g. WebCamTexture.didUpdateThisFrame or PassthroughCameraAccess.IsUpdatedThisFrame).
    /// When finished, invokes the callback with computed statistics.
    /// </summary>
    public void StartRecording(float durationSec, Func<bool> cameraUpdated, Action<FrameTimeStats> callback)
    {
        if (isRecording) return;

        frameTimes.Clear();
        remainingTime = durationSec;
        cameraUpdatedCheck = cameraUpdated;
        onComplete = callback;
        lastCameraUpdateTime = 0f;
        hasPreviousUpdate = false;
        isRecording = true;
    }

    private void Update()
    {
        if (!isRecording) return;

        remainingTime -= Time.unscaledDeltaTime;

        // If no camera-update delegate is set, treat every Unity frame as an update (fallback).
        bool updated = cameraUpdatedCheck != null ? cameraUpdatedCheck() : true;
        if (updated)
        {
            float now = Time.unscaledTime;
            if (hasPreviousUpdate)
            {
                float intervalMs = (now - lastCameraUpdateTime) * 1000f;
                frameTimes.Add(intervalMs);
            }
            lastCameraUpdateTime = now;
            hasPreviousUpdate = true;
        }

        if (remainingTime <= 0f)
        {
            isRecording = false;
            var localFrameTimes = new List<float>(frameTimes);
            StartCoroutine(ComputeAndFinish(localFrameTimes));
        }
    }

    private IEnumerator ComputeAndFinish(List<float> frames)
    {
        var computeTask = Task.Run(() => ComputeStats(frames));
        
        while (!computeTask.IsCompleted)
            yield return null;

        onComplete?.Invoke(computeTask.Result);
        onComplete = null;
        cameraUpdatedCheck = null;
    }

    private FrameTimeStats ComputeStats(List<float> frames)
    {
        int count = frames.Count;
        if (count == 0)
        {
            return new FrameTimeStats();
        }

        float sum = 0f;
        float min = float.MaxValue;
        float max = float.MinValue;

        for (int i = 0; i < count; i++)
        {
            float ft = frames[i];
            sum += ft;
            if (ft < min) min = ft;
            if (ft > max) max = ft;
        }

        float avg = sum / count;

        // Standard deviation
        float sqDiffSum = 0f;
        for (int i = 0; i < count; i++)
        {
            float diff = frames[i] - avg;
            sqDiffSum += diff * diff;
        }
        float sd = Mathf.Sqrt(sqDiffSum / count);

        // 95th percentile
        var sorted = new List<float>(frames);
        sorted.Sort();
        int p95Index = Mathf.CeilToInt(count * 0.95f) - 1;
        p95Index = Mathf.Clamp(p95Index, 0, count - 1);
        float p95 = sorted[p95Index];

        return new FrameTimeStats
        {
            AvgFrameTimeMs = avg,
            SdFrameTimeMs = sd,
            MinFrameTimeMs = min,
            MaxFrameTimeMs = max,
            P95FrameTimeMs = p95,
            TotalFrames = count,
            RecordingDurationSec = sum / 1000f
        };
    }
}
