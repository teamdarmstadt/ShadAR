using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEngine;

/// <summary>
/// Application logger writing to a file in persistentDataPath and the Unity console. The
/// head-locked debug text shows a single status line instead of the log: the step a prompt is in
/// (<see cref="SetStatus"/>), replaced by the latest warning or error. Safe to call from any
/// thread: calls made off the main thread are queued and applied on the next frame (TextMeshPro
/// must only be touched on the main thread).
/// </summary>
public class Logger : MonoBehaviour
{
    public enum LogLevel { Debug, Info, Warning, Error, None }

    [Header("Logging Settings")]
    [SerializeField] private bool logToFile = true;
    [Tooltip("Name of the log file in persistentDataPath")]
    [SerializeField] private string logFileName = "log.txt";
    [SerializeField] private bool logToConsole = true;
    [SerializeField] private bool logToDebugText = true;

    [SerializeField] private TMPro.TMP_Text debugText;

    [Tooltip("Minimum log level to record (messages below this level are ignored)")]
    [SerializeField] private LogLevel minimumLogLevel = LogLevel.Info;

    private readonly ConcurrentQueue<(string formatted, string message, LogLevel level)> offThreadLines = new();
    private readonly ConcurrentQueue<(string text, bool busy)> offThreadStatus = new();
    private string status = "";
    private float busySince = -1f; // >= 0 while a step runs; the status then counts its seconds
    private int shownSeconds;
    private string logFilePath;
    private int mainThreadId;

    private void Awake()
    {
        mainThreadId = Thread.CurrentThread.ManagedThreadId;
        logFilePath = Path.Combine(Application.persistentDataPath, logFileName);

        if (!logToFile) return;
        try
        {
            File.AppendAllText(logFilePath, $"\n[{DateTime.Now}] Logger initialized.\n");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[Logger] Failed to initialize log file: {ex.Message}");
            logToFile = false;
        }
    }

    private void Update()
    {
        while (offThreadLines.TryDequeue(out var line))
        {
            WriteLog(line.formatted, line.message, line.level);
        }
        while (offThreadStatus.TryDequeue(out var update))
        {
            ApplyStatus(update.text, update.busy);
        }

        if (busySince >= 0f && (int)(Time.time - busySince) != shownSeconds)
        {
            shownSeconds = (int)(Time.time - busySince);
            ShowStatus();
        }
    }

    /// <summary>
    /// Replaces the on-screen status line. A busy step counts its seconds up, so a long wait (the
    /// bundle compile takes over a minute) visibly progresses instead of looking hung.
    /// </summary>
    public void SetStatus(string text, bool busy = false)
    {
        if (Thread.CurrentThread.ManagedThreadId == mainThreadId)
        {
            ApplyStatus(text, busy);
        }
        else
        {
            offThreadStatus.Enqueue((text, busy));
        }
    }

    private void ApplyStatus(string text, bool busy)
    {
        status = text;
        busySince = busy ? Time.time : -1f;
        shownSeconds = 0;
        ShowStatus();
    }

    private void ShowStatus()
    {
        if (!logToDebugText || debugText == null) return;

        // DebugTextBg auto-sizes to the wrapped line count (VerticalLayoutGroup +
        // ContentSizeFitter), so the strip follows the text by itself.
        debugText.text = busySince >= 0f ? $"{status} {shownSeconds} s" : status;
    }

    public void Log(string message, string callerName = "", LogLevel level = LogLevel.Info)
    {
        if (level < minimumLogLevel || minimumLogLevel == LogLevel.None)
            return;

        string formatted = string.IsNullOrEmpty(callerName)
            ? $"[{level}] [{DateTime.Now}] {message}"
            : $"[{level}] [{DateTime.Now}] [{callerName}] {message}";

        if (Thread.CurrentThread.ManagedThreadId == mainThreadId)
        {
            WriteLog(formatted, message, level);
        }
        else
        {
            offThreadLines.Enqueue((formatted, message, level));
        }
    }

    public void LogDebug(string message, string callerName = "") => Log(message, callerName, LogLevel.Debug);

    public void LogInfo(string message, string callerName = "") => Log(message, callerName, LogLevel.Info);

    public void LogWarning(string message, string callerName = "") => Log(message, callerName, LogLevel.Warning);

    public void LogError(string message, string callerName = "") => Log(message, callerName, LogLevel.Error);

    private void WriteLog(string formattedMessage, string message, LogLevel level)
    {
        if (logToConsole)
        {
            switch (level)
            {
                case LogLevel.Warning: Debug.LogWarning(formattedMessage); break;
                case LogLevel.Error: Debug.LogError(formattedMessage); break;
                default: Debug.Log(formattedMessage); break;
            }
        }

        if (level >= LogLevel.Warning)
        {
            ApplyStatus($"{level}: {message}", false);
        }

        if (logToFile)
        {
            try
            {
                File.AppendAllText(logFilePath, formattedMessage + "\n");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[Logger] Failed to write to log file: {ex.Message}");
                logToFile = false;
            }
        }
    }
}
