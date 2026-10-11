using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

/// <summary>
/// Holds all data for a single pipeline benchmark run.
/// Mode-agnostic.
/// Timestamps are stored as UTC epoch milliseconds for precise duration computation.
/// </summary>
public class BenchmarkSession
{
    public string RequestId { get; }
    public string Mode { get; }
    /// <summary>
    /// The shader history session folder name (e.g. "shader_20260308_112500").
    /// Set automatically from ShaderHistorySession once available.
    /// </summary>
    public string SessionId { get; set; }
    public string GptReasoning { get; set; }
    public string ShaderName { get; set; }
    public FrameTimeStats FrameTimeStats { get; set; }

    private readonly List<(string name, long epochMs)> timestamps = new();

    public BenchmarkSession(string mode)
    {
        RequestId = DateTime.Now.ToString("yyyyMMddHHmm");
        Mode = mode;
    }

    /// <summary>
    /// Records a named pipeline stage with the current UTC time.
    /// Duplicate stage names are silently ignored (only the first call wins).
    /// </summary>
    public void RecordTimestamp(string stageName)
    {
        // Prevent duplicates caused by multiple subscribers firing the same pipeline stage
        for (int i = 0; i < timestamps.Count; i++)
        {
            if (timestamps[i].name == stageName) return;
        }

        long epochMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        timestamps.Add((stageName, epochMs));
    }

    /// <summary>
    /// Serializes the session to pretty-printed JSON.
    /// </summary>
    public string ToJson()
    {
        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder(1024);

        sb.AppendLine("{");
        sb.AppendLine($"  \"request_id\": \"{RequestId}\",");
        sb.AppendLine($"  \"mode\": \"{Escape(Mode)}\",");
        sb.AppendLine($"  \"gpt_reasoning\": \"{Escape(GptReasoning ?? "none")}\",");
        sb.AppendLine($"  \"session_id\": \"{Escape(SessionId ?? "unknown")}\",");
        sb.AppendLine($"  \"shader_name\": \"{Escape(ShaderName ?? "unknown")}\",");

        // --- timestamps ---
        sb.Append("  \"timestamps\": {");
        for (int i = 0; i < timestamps.Count; i++)
        {
            var (name, epochMs) = timestamps[i];
            sb.Append($"\n    \"time_{i}_{name}\": {epochMs}");
            if (i < timestamps.Count - 1) sb.Append(',');
        }
        if (timestamps.Count > 0) sb.AppendLine();
        else sb.AppendLine();
        sb.AppendLine("  },");

        // --- durations between consecutive stages ---
        sb.Append("  \"durations_ms\": {");
        bool hasDurations = false;
        for (int i = 1; i < timestamps.Count; i++)
        {
            if (hasDurations) sb.Append(',');
            long delta = timestamps[i].epochMs - timestamps[i - 1].epochMs;
            sb.Append($"\n    \"{timestamps[i - 1].name}_to_{timestamps[i].name}\": {delta}");
            hasDurations = true;
        }
        if (timestamps.Count >= 2)
        {
            long total = timestamps[timestamps.Count - 1].epochMs - timestamps[0].epochMs;
            sb.Append($",\n    \"total\": {total}");
        }
        if (hasDurations || timestamps.Count >= 2) sb.AppendLine();
        else sb.AppendLine();
        sb.AppendLine("  },");

        // --- frame time stats ---
        if (FrameTimeStats != null)
        {
            var s = FrameTimeStats;
            sb.AppendLine("  \"frame_time_stats\": {");
            sb.AppendLine($"    \"avg_frame_time_ms\": {s.AvgFrameTimeMs.ToString("F2", inv)},");
            sb.AppendLine($"    \"sd_frame_time_ms\": {s.SdFrameTimeMs.ToString("F2", inv)},");
            sb.AppendLine($"    \"min_frame_time_ms\": {s.MinFrameTimeMs.ToString("F2", inv)},");
            sb.AppendLine($"    \"max_frame_time_ms\": {s.MaxFrameTimeMs.ToString("F2", inv)},");
            sb.AppendLine($"    \"p95_frame_time_ms\": {s.P95FrameTimeMs.ToString("F2", inv)},");
            sb.AppendLine($"    \"total_frames\": {s.TotalFrames},");
            sb.AppendLine($"    \"recording_duration_sec\": {s.RecordingDurationSec.ToString("F2", inv)}");
            sb.AppendLine("  }");
        }
        else
        {
            sb.AppendLine("  \"frame_time_stats\": null");
        }

        sb.Append("}");
        return sb.ToString();
    }

    private static string Escape(string s)
    {
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "");
    }
}

/// <summary>
/// Aggregated frame time statistics computed after a recording window.
/// </summary>
public class FrameTimeStats
{
    public float AvgFrameTimeMs;
    public float SdFrameTimeMs;
    public float MinFrameTimeMs;
    public float MaxFrameTimeMs;
    public float P95FrameTimeMs;
    public int TotalFrames;
    public float RecordingDurationSec;
}
