using System;
using System.IO;
using UnityEngine;

/// <summary>
/// Singleton MonoBehaviour that orchestrates pipeline benchmark logging.
/// Call BeginSession / RecordTimestamp / OnResultApplied from pipeline components.
/// Each run is named after its shader-history folder (e.g. "shader_20260308_112500").
///
/// Integration flow:
///   WhisperTranscriber  → BeginSession + RecordTimestamp("speech_complete")
///   WhisperTranscriber  → RecordTimestamp("audio_sent_to_whisper")
///   GenerativeShaderController → SetSessionId
///   LLMShaderGenerator  → RecordTimestamp("sent_to_openai" | "sent_to_claude") + SetReasoningEffort
///   ShaderBundleWSClient→ RecordTimestamp("script_sent_to_quest_packer")
///   ShaderBundleWSClient→ RecordTimestamp("asset_bundle_received")
///   ShaderOverlayLoader → OnResultApplied(shaderName)  [records "shader_applied" + starts FPS benchmark]
/// </summary>
public class PipelineBenchmarkLogger : MonoBehaviour
{
    private const string LogFolderName = "benchmark_logs";

    public static PipelineBenchmarkLogger Instance { get; private set; }

    [Header("Dependencies")]
    [SerializeField] private FrameTimeBenchmark frameTimeBenchmark;
    [SerializeField] private ShaderOverlayLoader shaderOverlayLoader;
    [SerializeField] private Logger logger;

    [Header("Settings")]
    [Tooltip("Duration in seconds to record frame times after a result is applied")]
    [SerializeField] private float frameRecordDuration = 5f;

    [Tooltip("Enable or disable benchmark logging entirely")]
    [SerializeField] private bool enableLogging = true;

    private BenchmarkSession currentSession;
    private string logFolderPath;

    // ------------------------------------------------------------------ lifecycle

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;

        logFolderPath = Path.Combine(Application.persistentDataPath, LogFolderName);
        Directory.CreateDirectory(logFolderPath);
    }

    private void OnDestroy()
    {
        if (Instance == this) Instance = null;
    }

    // ------------------------------------------------------------------ public API

    /// <summary>
    /// Starts a new benchmark session. If a previous session was not finalized it is discarded.
    /// </summary>
    public void BeginSession()
    {
        if (!enableLogging) return;

        if (currentSession != null)
        {
            logger?.LogWarning("Previous benchmark session was not finalized. Discarding.", nameof(PipelineBenchmarkLogger));
        }

        currentSession = new BenchmarkSession("GenerativeShader");
    }

    /// <summary>
    /// Names the current run after its shader-history folder.
    /// </summary>
    public void SetSessionId(string sessionFolderName)
    {
        if (currentSession != null) currentSession.SessionId = sessionFolderName;
    }

    /// <summary>
    /// Records a timestamp for a named pipeline stage.
    /// Safe to call when no session is active (becomes a no-op).
    /// </summary>
    public void RecordTimestamp(string stageName)
    {
        currentSession?.RecordTimestamp(stageName);
    }

    /// <summary>
    /// Sets the GPT reasoning effort level metadata for the current session.
    /// </summary>
    public void SetReasoningEffort(string effort)
    {
        if (currentSession != null) currentSession.GptReasoning = effort;
    }

    /// <summary>
    /// Called when the pipeline's final visual result is applied (shader or stream output).
    /// Records the final timestamp, sets the result name, and starts frame-time recording.
    /// After the recording window, writes the log to persistent storage.
    /// </summary>
    /// <param name="resultName">Display name of the applied result (e.g. shader name).</param>
    /// <param name="finalStageName">Stage name for the timestamp (default: "shader_applied").</param>
    public void OnResultApplied(string resultName, string finalStageName = "shader_applied")
    {
        if (currentSession == null) return;

        currentSession.RecordTimestamp(finalStageName);
        currentSession.ShaderName = resultName;

        logger?.LogDebug(
            $"Result applied: {resultName}. Recording frame times for {frameRecordDuration}s.",
            nameof(PipelineBenchmarkLogger));

        if (frameTimeBenchmark != null)
        {
            Func<bool> cameraCheck = shaderOverlayLoader != null
                ? shaderOverlayLoader.IsCameraUpdatedThisFrame
                : null;

            frameTimeBenchmark.StartRecording(frameRecordDuration, cameraCheck, stats =>
            {
                currentSession.FrameTimeStats = stats;
                WriteAndClose();
            });
        }
        else
        {
            WriteAndClose();
        }
    }

    /// <summary>
    /// Whether a benchmark session is currently in progress.
    /// </summary>
    public bool HasActiveSession => currentSession != null;

    // ------------------------------------------------------------------ internals

    private void WriteAndClose()
    {
        if (currentSession == null) return;

        BenchmarkLogWriter.WriteLog(logFolderPath, currentSession, logger);
        logger?.LogInfo($"Benchmark log written for session {currentSession.SessionId ?? currentSession.RequestId}",
            nameof(PipelineBenchmarkLogger));
        currentSession = null;
    }

}
