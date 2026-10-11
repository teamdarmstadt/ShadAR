using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Records object-detection round-trip latency for the performance evaluation.
///
/// "RTT" here is the wall-clock time from the moment a frame is transmitted from
/// the Quest to the detection server until its labels + segmentation arrive back
/// and are parsed — i.e. the full send→receive cost, not the server-side compute
/// time alone. <see cref="Networking.DetectionNetworkClient"/> measures it with a
/// high-resolution <see cref="System.Diagnostics.Stopwatch"/> timestamp captured
/// at send and push the result here via <see cref="Record"/>.
///
/// Recording is gated by a controller button (Inspector-assignable). Hold the
/// button to record: each hold becomes one group with an incrementing group id.
/// Every detection that arrives while the button is held is written as one CSV
/// row (one row per label). The whole app run shares a single CSV file:
///
///     timestamp,label,rtt_ms,group_id
///
/// Rows are flushed immediately so data survives an app kill on the headset.
/// <see cref="Record"/> is thread-safe and a no-op when not recording, so the
/// hot path stays cheap when evaluation is off.
/// </summary>
public class DetectionLatencyBenchmark : MonoBehaviour
{
    public static DetectionLatencyBenchmark Instance { get; private set; }

    [Header("Recording Trigger")]
    [Tooltip("Hold this controller button to record. Each hold is one group.")]
    [SerializeField] private OVRInput.Button recordButton = OVRInput.Button.Two;

    [Tooltip("Which controller the record button is read from.")]
    [SerializeField] private OVRInput.Controller recordController = OVRInput.Controller.RTouch;

    [Header("Output")]
    [Tooltip("Subfolder under persistentDataPath where the CSV is written.")]
    [SerializeField] private string outputFolderName = "detection_latency";

    [Header("Debug")]
    [SerializeField] private Logger logger;
    [SerializeField] private bool logEvents = true;

    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    // gate guards every field below it — Record() may be called from a network
    // thread (UDP) while the button is polled on the main thread.
    private readonly object gate = new object();
    private StreamWriter writer;
    private string filePath;
    private bool recording;
    private int currentGroupId;
    private int rowsInCurrentGroup;

    /// <summary>Whether a recording group is currently active.</summary>
    public bool IsRecording { get { lock (gate) return recording; } }

    /// <summary>Absolute path of the CSV file (null until the first group starts).</summary>
    public string FilePath { get { lock (gate) return filePath; } }

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(this);
            return;
        }
        Instance = this;

        if (logger == null) logger = FindAnyObjectByType<Logger>();
    }

    private void Update()
    {
        // Poll the controller on the main thread only.
        if (OVRInput.GetDown(recordButton, recordController))
        {
            StartGroup();
        }
        else if (OVRInput.GetUp(recordButton, recordController))
        {
            StopGroup();
        }
    }

    private void StartGroup()
    {
        lock (gate)
        {
            if (recording) return;
            EnsureWriter();
            if (writer == null) return; // file could not be opened

            currentGroupId++;
            rowsInCurrentGroup = 0;
            recording = true;
        }

        if (logEvents)
            logger?.LogInfo($"Detection-latency recording started (group {currentGroupId}).",
                nameof(DetectionLatencyBenchmark));
    }

    private void StopGroup()
    {
        int group, rows;
        lock (gate)
        {
            if (!recording) return;
            recording = false;
            group = currentGroupId;
            rows = rowsInCurrentGroup;
            writer?.Flush();
        }

        if (logEvents)
            logger?.LogInfo($"Detection-latency recording stopped (group {group}, {rows} rows).",
                nameof(DetectionLatencyBenchmark));
    }

    /// <summary>
    /// Records one detection round-trip. Writes one CSV row per detected label,
    /// all sharing the same arrival timestamp, RTT and group id. Thread-safe and
    /// a no-op when not currently recording.
    /// </summary>
    /// <param name="rttMs">Round-trip time in milliseconds (Quest send → labels/mask received).</param>
    /// <param name="detections">The labels/segmentation that arrived for this frame.</param>
    public void Record(float rttMs, IReadOnlyList<DetectedObject> detections)
    {
        if (detections == null || detections.Count == 0) return;

        lock (gate)
        {
            if (!recording || writer == null) return;

            // Single arrival timestamp shared by every label in this frame.
            string ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff", Inv);
            string rtt = rttMs.ToString("F2", Inv);
            string group = currentGroupId.ToString(Inv);

            for (int i = 0; i < detections.Count; i++)
            {
                writer.Write(ts);
                writer.Write(',');
                writer.Write(CsvEscape(detections[i].Label));
                writer.Write(',');
                writer.Write(rtt);
                writer.Write(',');
                writer.Write(group);
                writer.Write('\n');
                rowsInCurrentGroup++;
            }

            // Flush per response so an app kill on the headset never loses a group.
            writer.Flush();
        }
    }

    private void EnsureWriter()
    {
        if (writer != null) return;

        try
        {
            string folder = Path.Combine(Application.persistentDataPath, outputFolderName);
            Directory.CreateDirectory(folder);

            // One CSV per app run, timestamped at first use.
            string fileName = $"detection_latency_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
            filePath = Path.Combine(folder, fileName);

            var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.Read);
            writer = new StreamWriter(stream, new UTF8Encoding(false));
            writer.WriteLine("timestamp,label,rtt_ms,group_id");
            writer.Flush();

            if (logEvents)
                logger?.LogInfo($"Detection-latency CSV: {filePath}", nameof(DetectionLatencyBenchmark));
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to open detection-latency CSV: {ex.Message}",
                nameof(DetectionLatencyBenchmark));
            writer = null;
        }
    }

    private static string CsvEscape(string value)
    {
        if (string.IsNullOrEmpty(value)) return "";
        if (value.IndexOf(',') < 0 && value.IndexOf('"') < 0 && value.IndexOf('\n') < 0)
            return value;
        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private void OnApplicationPause(bool paused)
    {
        if (paused) StopGroup(); // headset taken off / app backgrounded
    }

    private void OnDisable() => StopGroup();

    private void OnDestroy()
    {
        StopGroup();
        lock (gate)
        {
            writer?.Dispose();
            writer = null;
        }
        if (Instance == this) Instance = null;
    }
}
