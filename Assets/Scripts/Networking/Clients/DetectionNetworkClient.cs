using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Networking
{
    /// <summary>
    /// Mask encoding requested from the detection server.
    /// </summary>
    public enum MaskFormat
    {
        None = 0,
        Raster = 1,
        Polygon = 2,
        Rle = 3
    }

    /// <summary>
    /// UDP control packet that configures the detector: the active shader's open-vocabulary
    /// classes (slot order = label id) plus an optional detector switch.
    /// </summary>
    [Serializable]
    public class VocabularyConfigPayload
    {
        public List<string> classes = new();
        public float confidence = -1f;

        /// <summary>
        /// Detector switch ("yoloe" | "yolo"), applied by the server before the classes.
        /// Empty = leave the detector alone.
        /// </summary>
        public string detector = "";

        /// <summary>
        /// COCO class per slot (same length/order as <see cref="classes"/>, empty where a slot has
        /// no equivalent), written by the LLM into the shader header. Only the closed-vocabulary
        /// detector reads it; an empty list makes the server fall back to name matching.
        /// </summary>
        public List<string> coco_classes = new();
    }

    /// <summary>
    /// Streams the left passthrough feed to the detection server over UDP and turns the binary
    /// responses into <see cref="DetectedObject"/>s (polygon masks rasterized on the GPU).
    /// One frame is in flight at a time, so the send rate adapts to the server's speed.
    /// </summary>
    public class DetectionNetworkClient : MonoBehaviour, IDetectionProvider
    {
        private const string DetectorPref = "PromptedVision.Detector";
        private const uint VocabularyAckFrameId = 0xFFFFFFFEu;
        private const int InFlightRingSize = 16; // power of two

        private struct SentFrameRecord
        {
            public uint frameId;
            public int generation;
            public long sendTicks;
        }

        [Header("Detection Settings")]
        [SerializeField] private float scoreThreshold = 0.25f;
        [SerializeField] private MaskFormat maskFormat = MaskFormat.Polygon;
        [SerializeField] private bool requestSegmentationMasks = true;
        [SerializeField] [Range(64, 1024)] private int maskResolution = 256;
        [SerializeField] [Range(16, 256)] private int maxPolygonPoints = 64;

        [Tooltip("Seconds to wait for a response before the next frame is sent anyway (lost packet).")]
        [SerializeField] private float responseTimeout = 1f;

        [Header("Processing")]
        [SerializeField] private ComputeShader polygonRasterizerShader;
        [SerializeField] private DetectionTracker detectionTracker;
        [SerializeField] private CameraScreenshotProvider screenshotProvider;
        [SerializeField] private Logger logger;

        public List<DetectedObject> CurrentDetections { get; private set; } = new();
        public event Action<List<DetectedObject>> OnDetectionsUpdated;

        private UdpImageTransmitter udpTransmitter;
        private GPUPolygonRasterizer gpuRasterizer;
        private Coroutine streamingCoroutine;

        // Handed over from the UDP receive thread to the main thread.
        private readonly object lockObj = new();
        private readonly SentFrameRecord[] sentFrames = new SentFrameRecord[InFlightRingSize];
        private readonly byte[] frameMetaBuffer = new byte[10];
        private readonly List<Texture> ownedMaskTextures = new();
        private List<DetectedObject> pendingDetections;
        private bool hasUpdate;
        private uint lastAcceptedFrameId;

        private volatile bool waitingForResponse;
        private float lastFrameTime;

        // A label_id is a slot of the vocabulary the frame was detected under. Results of a frame
        // sent before the last vocabulary change would light up the new shader's slots with the
        // old shader's objects, so each sent frame records its vocabularyGeneration and responses
        // from older generations are dropped.
        private volatile int vocabularyGeneration;
        private volatile bool vocabularyAckLogged;

        // Active shader's class list: label_id is the slot index into it. Assigned on the main
        // thread, read on the UDP thread; swapped by reference.
        private volatile string[] vocabulary;

        // Last configuration, re-sent whenever the socket (re)connects: UDP may drop it, and a
        // restarted server falls back to its command-line detector.
        private List<string> lastVocabulary;
        private List<string> lastCocoClasses;
        private float lastVocabularyConfidence = -1f;
        private string lastDetector = "";

        private ServerConnectionConfig Config => ServerConnectionConfig.Instance;

        /// <summary>Last detector requested by the user; empty = leave the server's default.</summary>
        public string ActiveDetector => lastDetector;

        private void Awake()
        {
            lastDetector = PlayerPrefs.GetString(DetectorPref, "");

            if (logger == null) logger = FindFirstObjectByType<Logger>();
            if (screenshotProvider == null) screenshotProvider = FindFirstObjectByType<CameraScreenshotProvider>();
            if (detectionTracker == null) detectionTracker = GetComponent<DetectionTracker>();
        }

        public void StartDetection()
        {
            if (udpTransmitter == null)
            {
                gpuRasterizer ??= new GPUPolygonRasterizer(polygonRasterizerShader, maxPolygonPoints);

                lock (lockObj)
                {
                    Array.Clear(sentFrames, 0, sentFrames.Length);
                    lastAcceptedFrameId = 0;
                }
                waitingForResponse = false;

                udpTransmitter = new UdpImageTransmitter();
                udpTransmitter.OnDataReceived += HandleUdpResponse;
                udpTransmitter.OnError += msg => logger?.LogWarning($"UDP: {msg}", nameof(DetectionNetworkClient));
                udpTransmitter.Connect(Config.Settings.detectionServerHost, Config.Settings.detectionServerUdpPort);
            }

            if (streamingCoroutine != null) StopCoroutine(streamingCoroutine);
            streamingCoroutine = StartCoroutine(StreamingLoop());

            // Covers a SetVocabulary that arrived before the socket existed.
            SendVocabularyUdp();
        }

        public void StopDetection()
        {
            if (streamingCoroutine != null)
            {
                StopCoroutine(streamingCoroutine);
                streamingCoroutine = null;
            }

            udpTransmitter?.Dispose();
            udpTransmitter = null;

            ReleaseOwnedMaskTextures(destroyAll: true);
            gpuRasterizer?.Dispose();
            gpuRasterizer = null;
        }

        private void OnDestroy() => StopDetection();

        private IEnumerator StreamingLoop()
        {
            while (true)
            {
                bool responseReadyOrTimedOut = !waitingForResponse || Time.time - lastFrameTime >= responseTimeout;
                // Check CanSendFrame BEFORE capturing and JPEG-encoding: previously, when server
                // RTT was faster than MinSendInterval (33 ms), SendFrame silently dropped the frame
                // while StreamingLoop set waitingForResponse = true, stalling for responseTimeout (1s).
                if (responseReadyOrTimedOut && screenshotProvider != null
                    && udpTransmitter != null && udpTransmitter.CanSendFrame)
                {
                    bool captured = false;
                    screenshotProvider.CaptureStreamingScreenshot(tex =>
                    {
                        if (tex != null && udpTransmitter != null && SendUdpFrame(tex))
                        {
                            waitingForResponse = true;
                            lastFrameTime = Time.time;
                        }
                        captured = true;
                    });

                    float waited = 0f;
                    while (!captured && waited < 1f)
                    {
                        yield return null;
                        waited += Time.deltaTime;
                    }
                }

                yield return null;
            }
        }

        public void SetVocabulary(IReadOnlyList<string> classes, float confidence = -1f,
                                  IReadOnlyList<string> cocoClasses = null)
        {
            var newClasses = classes != null ? new List<string>(classes) : new List<string>();

            // Only kept when it lines up slot-for-slot; a shifted list would point slots at the
            // wrong COCO classes, which is worse than letting the server match names.
            var newCocoClasses = cocoClasses != null && cocoClasses.Count == newClasses.Count
                ? new List<string>(cocoClasses)
                : null;

            // The shader menu previews a tile's vocabulary and then activates the same one.
            if (lastVocabulary != null
                && newClasses.SequenceEqual(lastVocabulary)
                && confidence == lastVocabularyConfidence
                && (newCocoClasses ?? new List<string>()).SequenceEqual(lastCocoClasses ?? new List<string>()))
            {
                return;
            }

            lastVocabulary = newClasses;
            lastVocabularyConfidence = confidence;
            lastCocoClasses = newCocoClasses;

            vocabulary = lastVocabulary.Count > 0 ? lastVocabulary.ToArray() : null;

            if (confidence > 0f)
            {
                scoreThreshold = confidence;
            }

            lock (lockObj)
            {
                vocabularyGeneration++;
                pendingDetections = new List<DetectedObject>();
                hasUpdate = true;
            }

            logger?.LogInfo($"Detection vocabulary: [{string.Join(", ", lastVocabulary)}] conf={confidence}", nameof(DetectionNetworkClient));
            SendVocabularyUdp();
        }

        /// <summary>
        /// Switches the server-side detector ("yoloe" open-vocab / "yolo" closed COCO). Persisted,
        /// and re-sent on every (re)connect since the server resets to its command-line default
        /// when it restarts.
        /// </summary>
        public void SetDetector(string detector)
        {
            lastDetector = string.IsNullOrWhiteSpace(detector) ? "" : detector.Trim().ToLowerInvariant();
            PlayerPrefs.SetString(DetectorPref, lastDetector);

            logger?.LogInfo($"Detector: '{lastDetector}'", nameof(DetectionNetworkClient));
            SendVocabularyUdp();
        }

        /// <summary>
        /// Sends vocabulary + detector as one control packet over the frame socket (sentinel frame
        /// id 0xFFFFFFFF), three times as cheap insurance against drops.
        /// </summary>
        private void SendVocabularyUdp()
        {
            if (udpTransmitter == null || !udpTransmitter.IsConnected) return;

            // A detector switch is worth sending on its own: the user can flip it before any
            // shader is active.
            if (lastVocabulary == null && string.IsNullOrEmpty(lastDetector)) return;

            var payload = new VocabularyConfigPayload
            {
                confidence = lastVocabularyConfidence,
                detector = lastDetector,
            };
            if (lastCocoClasses != null)
            {
                payload.coco_classes.AddRange(lastCocoClasses);
            }
            if (lastVocabulary != null)
            {
                foreach (var c in lastVocabulary)
                {
                    if (!string.IsNullOrWhiteSpace(c)) payload.classes.Add(c.Trim());
                }
            }

            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(JsonUtility.ToJson(payload));
            vocabularyAckLogged = false;
            for (int i = 0; i < 3; i++)
            {
                udpTransmitter.SendControl(bytes);
            }
        }

        private void Update()
        {
            if (!hasUpdate) return;

            List<DetectedObject> dets;
            lock (lockObj)
            {
                dets = pendingDetections;
                hasUpdate = false;
            }
            if (dets == null) return;

            // Return previous frame's mask RenderTextures to the pool before rasterizing the new
            // frame so pooled RenderTextures are immediately reused instead of leaking VRAM.
            ReleaseOwnedMaskTextures(destroyAll: false);

            if (gpuRasterizer != null && requestSegmentationMasks)
            {
                for (int i = 0; i < dets.Count; i++)
                {
                    var det = dets[i];
                    if (det.HasPolygon)
                    {
                        det.MaskTexture = gpuRasterizer.RasterizePolygon(det.MaskPolygon, maskResolution);
                        det.HasMask = det.MaskTexture != null;
                        if (det.MaskTexture != null)
                        {
                            ownedMaskTextures.Add(det.MaskTexture);
                        }
                        dets[i] = det;
                    }
                }
            }

            if (detectionTracker != null)
            {
                detectionTracker.UpdateFromInference(dets);
                CurrentDetections = detectionTracker.CurrentFrameDetections;
            }
            else
            {
                CurrentDetections = dets;
            }

            OnDetectionsUpdated?.Invoke(CurrentDetections);
        }

        private void ReleaseOwnedMaskTextures(bool destroyAll)
        {
            for (int i = 0; i < ownedMaskTextures.Count; i++)
            {
                var tex = ownedMaskTextures[i];
                if (tex == null) continue;

                if (!destroyAll && gpuRasterizer != null && tex is RenderTexture rt)
                {
                    gpuRasterizer.ReturnToPool(rt);
                }
                else
                {
                    if (tex is RenderTexture deadRt) deadRt.Release();
                    Destroy(tex);
                }
            }
            ownedMaskTextures.Clear();
        }

        private bool SendUdpFrame(Texture2D frame)
        {
            byte[] jpg = frame.EncodeToJPG(75);

            // [score:4][maskFormat:1][requestMasks:1][maskRes:2][maxPoints:2]
            WriteSingleLE(frameMetaBuffer, 0, scoreThreshold);
            frameMetaBuffer[4] = (byte)maskFormat;
            frameMetaBuffer[5] = (byte)(requestSegmentationMasks ? 1 : 0);
            WriteUInt16LE(frameMetaBuffer, 6, (ushort)maskResolution);
            WriteUInt16LE(frameMetaBuffer, 8, (ushort)maxPolygonPoints);

            long sendTicks = System.Diagnostics.Stopwatch.GetTimestamp();
            int gen = vocabularyGeneration;

            if (!udpTransmitter.TrySendFrame(jpg, frameMetaBuffer, out uint sentFrameId))
            {
                return false;
            }

            lock (lockObj)
            {
                sentFrames[sentFrameId & (InFlightRingSize - 1)] = new SentFrameRecord
                {
                    frameId = sentFrameId,
                    generation = gen,
                    sendTicks = sendTicks
                };
            }
            return true;
        }

        /// <summary>
        /// Runs on the UDP receive thread.
        /// </summary>
        private void HandleUdpResponse(byte[] data)
        {
            if (data == null || data.Length < 4) return;

            uint responseFrameId = BitConverter.ToUInt32(data, 0);

            // Control acks must NOT clear waitingForResponse for an in-flight detection frame.
            if (responseFrameId == VocabularyAckFrameId)
            {
                if (vocabularyAckLogged) return;
                vocabularyAckLogged = true;
                string ackJson = data.Length > 4 ? System.Text.Encoding.UTF8.GetString(data, 4, data.Length - 4) : "";
                logger?.LogInfo($"Server vocabulary ack: {ackJson}", nameof(DetectionNetworkClient));
                return;
            }

            // Header: [frame_id:4][num_detections:2]
            if (data.Length < 6) return;

            long matchedSendTicks = 0;
            lock (lockObj)
            {
                var record = sentFrames[responseFrameId & (InFlightRingSize - 1)];
                bool matchesInFlight = record.frameId == responseFrameId;

                // Only clear waitingForResponse when this response is at least as recent as the
                // latest accepted frame (protects against delayed duplicates from timed-out frames).
                if (lastAcceptedFrameId == 0 || (int)(responseFrameId - lastAcceptedFrameId) > 0)
                {
                    waitingForResponse = false;
                }
                else
                {
                    return; // Out-of-order or duplicate packet from an older frame
                }

                // Drop frames sent under a previous shader's vocabulary so old object slots never
                // light up the newly applied shader.
                if (!matchesInFlight || record.generation != vocabularyGeneration)
                {
                    return;
                }

                lastAcceptedFrameId = responseFrameId;
                matchedSendTicks = record.sendTicks;
            }

            int offset = 4;
            ushort numDetections = BitConverter.ToUInt16(data, offset); offset += 2;
            var detections = new List<DetectedObject>(numDetections);

            for (int i = 0; i < numDetections; i++)
            {
                // Detection: [label_id:2][x:4][y:4][w:4][h:4][conf:2][num_points:2][points:4*n]
                if (offset + 22 > data.Length) break;

                ushort labelId = BitConverter.ToUInt16(data, offset); offset += 2;
                float x = BitConverter.ToSingle(data, offset); offset += 4;
                float y = BitConverter.ToSingle(data, offset); offset += 4;
                float w = BitConverter.ToSingle(data, offset); offset += 4;
                float h = BitConverter.ToSingle(data, offset); offset += 4;
                float conf = BitConverter.ToUInt16(data, offset) / 65535f; offset += 2;
                ushort numPoints = BitConverter.ToUInt16(data, offset); offset += 2;

                Vector2[] polygon = null;
                if (numPoints > 0)
                {
                    if (offset + numPoints * 4 > data.Length) break;

                    polygon = new Vector2[numPoints];
                    for (int p = 0; p < numPoints; p++)
                    {
                        ushort px = BitConverter.ToUInt16(data, offset); offset += 2;
                        ushort py = BitConverter.ToUInt16(data, offset); offset += 2;
                        polygon[p] = new Vector2(px / 65535f, py / 65535f);
                    }
                }

                detections.Add(new DetectedObject
                {
                    Label = ResolveLabel(labelId),
                    LabelID = labelId,
                    // Image space has y down; Unity viewport space has y up.
                    BoundingBox = new Rect(x, 1.0f - y, w, h),
                    Confidence = conf,
                    MaskPolygon = polygon,
                    HasPolygon = polygon != null
                });
            }

            lock (lockObj)
            {
                if (vocabularyGeneration != sentFrames[responseFrameId & (InFlightRingSize - 1)].generation)
                {
                    return;
                }
                pendingDetections = detections;
                hasUpdate = true;
            }

            if (matchedSendTicks != 0 && DetectionLatencyBenchmark.Instance != null)
            {
                float rttMs = (float)((System.Diagnostics.Stopwatch.GetTimestamp() - matchedSendTicks)
                                      * 1000.0 / System.Diagnostics.Stopwatch.Frequency);
                DetectionLatencyBenchmark.Instance.Record(rttMs, detections);
            }
        }

        private string ResolveLabel(int labelId)
        {
            var vocab = vocabulary;
            return vocab != null && labelId < vocab.Length ? vocab[labelId] : $"Item {labelId}";
        }

        private static void WriteSingleLE(byte[] buffer, int offset, float value)
        {
            int bits = BitConverter.SingleToInt32Bits(value);
            buffer[offset] = (byte)bits;
            buffer[offset + 1] = (byte)(bits >> 8);
            buffer[offset + 2] = (byte)(bits >> 16);
            buffer[offset + 3] = (byte)(bits >> 24);
        }

        private static void WriteUInt16LE(byte[] buffer, int offset, ushort value)
        {
            buffer[offset] = (byte)value;
            buffer[offset + 1] = (byte)(value >> 8);
        }
    }
}
