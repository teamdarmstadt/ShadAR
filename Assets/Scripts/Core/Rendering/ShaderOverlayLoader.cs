using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using Meta.XR;
using Networking;

/// <summary>
/// GPU layout of one detection as seen by the generated shaders (StructuredBuffer _Detections).
/// Must match the struct declared in object_detection_prompt.txt — 56 bytes.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ShaderDetection
{
    public Vector4 boundingBox;    // x, y, width, height (normalized 0-1)
    public float confidence;
    public int labelID;            // slot index into the shader's @detect-classes list
    public Vector3 worldPosition;  // unused (detections are 2D); kept for the prompt's struct layout
    public float depth;            // unused, see above
    public int hasDepth;           // always 0
    public int hasMask;
    public int maskIndex;          // cell index into the mask atlas
    public float padding;
}

/// <summary>
/// Per-eye render target and GPU state.
/// </summary>
[System.Serializable]
public class EyeRenderData
{
    public PassthroughCameraAccess cameraAccess;
    public MeshRenderer meshRenderer;

    [System.NonSerialized] public Material defaultMaterial;
    [System.NonSerialized] public Material currentMaterial;
    [System.NonSerialized] public ComputeBuffer detectionBuffer;
    [System.NonSerialized] public Material uploadedMaterial;
    [System.NonSerialized] public int uploadedVersion = -1;
    [System.NonSerialized] public int uploadedDetectionCount;

    public void InvalidateUploadState()
    {
        uploadedMaterial = null;
        uploadedVersion = -1;
    }

    public void ReleaseBuffer()
    {
        detectionBuffer?.Release();
        detectionBuffer = null;
        InvalidateUploadState();
    }

    public Texture GetCameraTexture()
    {
        // enabled check: GetTexture() logs an error every call while the component is
        // disabled (e.g. after a failed Play) — return null quietly instead.
        return cameraAccess != null && cameraAccess.enabled ? cameraAccess.GetTexture() : null;
    }
}

/// <summary>
/// Renders the passthrough feed of each eye onto its quad, applies the active generated shader
/// and feeds it the per-eye detections (buffer + mask atlas) when detections change.
/// </summary>
public class ShaderOverlayLoader : MonoBehaviour
{
    // Eye-gated unlit shader: URP ignores Camera.stereoTargetEye, so ONE stereo camera renders
    // both quads and each quad's material collapses itself in the other eye via _EyeIndex.
    private const string DefaultUnlitShader = "PromptedVision/StereoEyeUnlit";
    private const string StereoRenderingPref = "PromptedVision.StereoRendering";

    private static readonly int EyeIndexProperty = Shader.PropertyToID("_EyeIndex");
    private static readonly int MainTextureProperty = Shader.PropertyToID("_MainTex");
    private static readonly int DetectionsBufferProperty = Shader.PropertyToID("_Detections");
    private static readonly int DetectionCountProperty = Shader.PropertyToID("_DetectionCount");
    private static readonly int MaskTextureProperty = Shader.PropertyToID("_MaskTex");
    private static readonly int HasMaskProperty = Shader.PropertyToID("_HasMask");
    private static readonly int MaskCountProperty = Shader.PropertyToID("_MaskCount");
    private static readonly int MaxMasksProperty = Shader.PropertyToID("_MaxMasks");
    private static readonly List<DetectedObject> NoDetections = new();

    [Header("Stereoscopic Eye Configuration")]
    [SerializeField] private EyeRenderData leftEye = new();
    [SerializeField] private EyeRenderData rightEye = new();

    [Tooltip("Stereo = each eye its own passthrough feed (true per-eye depth). Mono = both eyes show " +
             "the LEFT feed (zero disparity). Runtime-toggleable via SetStereoRendering(); persisted in " +
             "PlayerPrefs. The OpenXR render mode stays Single Pass Instanced either way.")]
    [SerializeField] private bool stereoRendering = true;

    [Header("Detection Source")]
    [Tooltip("Detection provider (DetectionNetworkClient). Its DetectionTracker projects the boxes per eye.")]
    [SerializeField] private MonoBehaviour detectionProviderSource;

    [SerializeField] private Logger logger;
    [SerializeField] private ShaderBundleWebSocketClient webSocketClient;

    [Header("Performance")]
    [Tooltip("Mask resolution per atlas cell (lower = faster, 256 recommended)")]
    [SerializeField] private int maskAtlasResolution = 256;

    [Header("Camera Startup Restart")]
    [Tooltip("Works around a Meta Passthrough Camera cold-start issue: after a fresh launch the " +
             "camera reports frames as 'updated' but the content stays black until a real Stop/Play " +
             "(what a manual headset sleep/wake does). Forces one Stop/Play at startup.")]
    [SerializeField] private bool forceStartupCameraRestart = true;

    [Tooltip("Seconds to let the camera start before the startup Stop/Play, and the verification " +
             "window after it.")]
    [SerializeField] private float cameraKickTimeoutSeconds = 1.5f;

    private ShaderBundleManager shaderBundleManager;
    // Mask textures are bounding-box-local and identical for both eyes, so one shared atlas
    // halves VRAM and avoids duplicating GPU blits/copies per inference frame.
    private OptimizedMaskAtlas maskAtlas;
    private int maskAtlasVersion = -1;
    private int[] cachedMaskIndices;
    private DetectionTracker detectionTracker;
    private PassthroughViewFitter rightEyeFitter;
    private ShaderDetection[] shaderDetectionsBuffer = new ShaderDetection[16];
    private readonly List<Texture> maskListBuffer = new(16);

    public IDetectionProvider DetectionProvider => detectionProviderSource as IDetectionProvider;

    /// <summary>True = per-eye stereo rendering; false = mono (both eyes show the left feed).</summary>
    public bool StereoRendering => stereoRendering;

    /// <summary>
    /// Fired when the active shader changes. The argument is the shader's history folder (from
    /// which its detection manifest is resolved), or null for plain passthrough.
    /// </summary>
    public event System.Action<string> OnActiveShaderBundleChanged;

    private void Start()
    {
        stereoRendering = PlayerPrefs.GetInt(StereoRenderingPref, stereoRendering ? 1 : 0) == 1;

        maskAtlas = new OptimizedMaskAtlas(maskAtlasResolution);
        detectionTracker = detectionProviderSource != null ? detectionProviderSource.GetComponent<DetectionTracker>() : null;
        rightEyeFitter = rightEye.meshRenderer != null ? rightEye.meshRenderer.GetComponent<PassthroughViewFitter>() : null;
        SyncRightEyeFitter();

        shaderBundleManager = new ShaderBundleManager(logger);
        StartCoroutine(AssignDefaultMaterials());

        // The detection client grabs its frames from the left feed through this provider.
        CameraScreenshotProvider.Instance?.RegisterCameraScreenshotProvider(GetCurrentCameraFeed);
    }

    private void OnEnable()
    {
        if (webSocketClient != null) webSocketClient.OnBundleReceived += OnBundleReceived;
    }

    private void OnDisable()
    {
        // Disabling stops the coroutines without running their finally blocks.
        scanning = false;
        if (webSocketClient != null) webSocketClient.OnBundleReceived -= OnBundleReceived;
    }

    private void OnDestroy()
    {
        leftEye.ReleaseBuffer();
        rightEye.ReleaseBuffer();
        if (rightEye.currentMaterial != null && rightEye.currentMaterial != rightEye.defaultMaterial)
        {
            Destroy(rightEye.currentMaterial);
        }
        if (leftEye.defaultMaterial != null) Destroy(leftEye.defaultMaterial);
        if (rightEye.defaultMaterial != null) Destroy(rightEye.defaultMaterial);
        maskAtlas?.Dispose();
    }

    private void Update()
    {
        UpdateEyeMaterial(leftEye, leftEye.cameraAccess);
        // Mono: the right eye mirrors the LEFT camera (feed + pose) → both eyes show the same
        // image with zero disparity. Stereo: each eye uses its own camera.
        UpdateEyeMaterial(rightEye, stereoRendering ? rightEye.cameraAccess : leftEye.cameraAccess);
    }

    /// <summary>Left passthrough feed (also the frame source for detection).</summary>
    public Texture GetCurrentCameraFeed() => leftEye.GetCameraTexture();

    /// <summary>True on frames where the left passthrough camera delivered a new image.</summary>
    public bool IsCameraUpdatedThisFrame() =>
        leftEye.cameraAccess != null && leftEye.cameraAccess.IsUpdatedThisFrame;

    /// <summary>
    /// Switches between per-eye stereo and mono (both eyes = left feed). Persisted in PlayerPrefs.
    /// Only the feed and right-quad optical alignment change, never the OpenXR render mode.
    /// </summary>
    public void SetStereoRendering(bool stereo)
    {
        if (stereoRendering == stereo) return;
        stereoRendering = stereo;
        rightEye.InvalidateUploadState();
        SyncRightEyeFitter();
        PlayerPrefs.SetInt(StereoRenderingPref, stereo ? 1 : 0);
        PlayerPrefs.Save();
        logger?.LogInfo($"Render mode -> {(stereo ? "Stereo (per-eye)" : "Mono (left feed both eyes)")}.",
            nameof(ShaderOverlayLoader));
    }

    private void SyncRightEyeFitter()
    {
        if (rightEyeFitter == null && rightEye.meshRenderer != null)
        {
            rightEyeFitter = rightEye.meshRenderer.GetComponent<PassthroughViewFitter>();
        }
        // In Mono mode, align QuadRight with the left camera's intrinsics/lens tilt relative to
        // RightEyeAnchor so displaying the left feed produces zero disparity instead of lens-tilt divergence.
        rightEyeFitter?.SetCameraOverride(stereoRendering ? null : leftEye.cameraAccess);
    }

    private void UpdateEyeMaterial(EyeRenderData eye, PassthroughCameraAccess cameraAccess)
    {
        var material = eye.currentMaterial;
        if (material == null) return;

        // Mono: the right eye reads the LEFT feed (matches the left pose passed in Update()).
        var sourceEye = (!stereoRendering && eye == rightEye) ? leftEye : eye;
        var feed = sourceEye.GetCameraTexture();
        if (feed != null && material.mainTexture != feed)
        {
            material.mainTexture = feed;
        }

        if (material != eye.defaultMaterial)
        {
            UpdateDetectionBuffer(material, cameraAccess, eye);
        }
    }

    // ------------------------------------------------------------------ startup

    private IEnumerator AssignDefaultMaterials()
    {
        // Gate on the LEFT camera only. The right camera may start late or fail; each eye still
        // gets its own material, so a dead right camera shows black on ITS eye instead of
        // silently mirroring the left feed.
        while (leftEye.GetCameraTexture() == null)
        {
            yield return null;
        }

        AssignDefaultMaterial(leftEye, 0f);
        AssignDefaultMaterial(rightEye, 1f);
        logger?.LogInfo("Passthrough feed bound to both eyes.", nameof(ShaderOverlayLoader));

        if (forceStartupCameraRestart)
        {
            StartCoroutine(PrimeCameraAtStartup(leftEye.cameraAccess, "LEFT"));
            if (rightEye.cameraAccess != null && rightEye.cameraAccess != leftEye.cameraAccess)
            {
                StartCoroutine(PrimeCameraAtStartup(rightEye.cameraAccess, "RIGHT"));
            }
        }
    }

    private void AssignDefaultMaterial(EyeRenderData eye, float eyeIndex)
    {
        if (eye.meshRenderer == null)
        {
            logger?.LogError("Eye MeshRenderer not assigned.", nameof(ShaderOverlayLoader));
            return;
        }

        var sourceEye = (!stereoRendering && eye == rightEye) ? leftEye : eye;
        eye.defaultMaterial = new Material(Shader.Find(DefaultUnlitShader));
        eye.defaultMaterial.SetFloat(EyeIndexProperty, eyeIndex);
        eye.defaultMaterial.mainTexture = sourceEye.GetCameraTexture(); // may be null; Update binds it later
        eye.currentMaterial = eye.defaultMaterial;
        eye.meshRenderer.material = eye.defaultMaterial;
    }

    /// <summary>
    /// Forces one Stop/Play cycle on a passthrough camera to clear the cold-start state where
    /// frames are reported as updated but the content stays black. Toggling enabled runs the
    /// camera's OnDisable (Stop) and OnEnable (Play); consumers re-fetch GetTexture() every
    /// frame, so the re-created texture is picked up automatically.
    /// </summary>
    private IEnumerator PrimeCameraAtStartup(PassthroughCameraAccess cam, string eyeName)
    {
        if (cam == null) yield break;

        yield return new WaitForSecondsRealtime(cameraKickTimeoutSeconds);

        // Permission still pending or component turned off — its own OnEnable will start it.
        if (!cam.enabled) yield break;

        cam.enabled = false;
        yield return null;
        cam.enabled = true;
        yield return null;

        float verifyDeadline = Time.realtimeSinceStartup + cameraKickTimeoutSeconds;
        while (Time.realtimeSinceStartup < verifyDeadline && !cam.IsUpdatedThisFrame)
        {
            yield return null;
        }

        logger?.LogInfo($"{eyeName} passthrough camera after startup restart: " +
                        $"{(cam.IsUpdatedThisFrame ? "frames flowing" : "no frames yet")}.",
            nameof(ShaderOverlayLoader));
    }

    // ------------------------------------------------------------------ shader history

    public string GetActiveShaderFolder() => FolderOf(shaderBundleManager?.GetCurrentBundlePath());

    public string GetShaderFolder(int index) => FolderOf(shaderBundleManager?.GetBundlePath(index));

    private static string FolderOf(string bundlePath) =>
        string.IsNullOrEmpty(bundlePath) ? null : System.IO.Path.GetDirectoryName(bundlePath);

    private void NotifyActiveBundleChanged() => OnActiveShaderBundleChanged?.Invoke(GetActiveShaderFolder());

    public void SwitchShader(int index)
    {
        shaderBundleManager.SwitchShader(index, ApplyShader);
        NotifyActiveBundleChanged();
    }

    public int GetShaderCount() => shaderBundleManager.GetShaderCount();

    public int GetCurrentShaderIndex() => shaderBundleManager.GetCurrentShaderIndex();

    public string GetShaderDisplayName(int index) => shaderBundleManager.GetShaderDisplayName(index);

    public string GetShaderName(int index) => shaderBundleManager.GetShaderName(index);

    /// <summary>Material of a history entry, for the menu's live previews.</summary>
    public Material GetShaderMaterial(int index) => shaderBundleManager?.GetShaderMaterial(index);

    /// <summary>
    /// Refreshes the shader history from disk without changing the active shader.
    /// </summary>
    public void RefreshShaderHistory(System.Action onComplete = null)
    {
        StartCoroutine(RefreshShaderHistoryCoroutine(onComplete));
    }

    private IEnumerator RefreshShaderHistoryCoroutine(System.Action onComplete)
    {
        yield return Exclusive(shaderBundleManager.ScanAndLoad(onLatestShaderApplied: null));
        onComplete?.Invoke();
    }

    private bool scanning;

    /// <summary>
    /// Runs history scans one after another. A received bundle triggers two of them (the
    /// auto-apply here and the history menu's refresh); interleaved, both rebuild the history and
    /// the menu's one restored the OLD selection last, so the new shader rendered but its
    /// detection classes were never sent and it looked like nothing had been applied.
    /// </summary>
    private IEnumerator Exclusive(IEnumerator scan)
    {
        while (scanning) yield return null;
        scanning = true;
        try { yield return scan; }
        finally { scanning = false; }
    }

    private void OnBundleReceived()
    {
        StartCoroutine(ScanLoadAndNotify());
    }

    private IEnumerator ScanLoadAndNotify()
    {
        yield return Exclusive(shaderBundleManager.ScanAndLoad(ApplyShader));
        NotifyActiveBundleChanged();
    }

    /// <summary>
    /// Binds the live detections to an arbitrary material (the history menu's selected tile).
    /// Reuses the left-eye ComputeBuffer and shared mask atlas without re-uploading GPU data
    /// when leftEye is already current for this detection version.
    /// </summary>
    public void BindCurrentDetectionsToPreview(Material material)
    {
        if (material == null) return;

        // Ensure leftEye.detectionBuffer and maskAtlas reflect the latest detections, then bind
        // the buffer/atlas uniforms onto the preview material without mutating leftEye.uploadedMaterial.
        EnsureEyeBufferData(leftEye.cameraAccess, leftEye);
        BindEyeResourcesToMaterial(material, leftEye);
    }

    // ------------------------------------------------------------------ applying shaders

    private void ApplyShader(Material mat)
    {
        ApplyShaderToEye(leftEye, mat, 0f);

        // The right eye needs its own instance (different _EyeIndex, feed and buffers).
        if (rightEye.currentMaterial != null && rightEye.currentMaterial != rightEye.defaultMaterial)
        {
            Destroy(rightEye.currentMaterial);
        }
        ApplyShaderToEye(rightEye, new Material(mat), 1f);

        PipelineBenchmarkLogger.Instance?.OnResultApplied(mat.shader.name);
    }

    private void ApplyShaderToEye(EyeRenderData eye, Material mat, float eyeIndex)
    {
        var sourceEye = (!stereoRendering && eye == rightEye) ? leftEye : eye;

        // Generated shaders must carry the _EyeIndex gate (see shader_prompt.txt); without it
        // SetFloat is a no-op and both quads draw into both eyes.
        mat.SetFloat(EyeIndexProperty, eyeIndex);
        mat.renderQueue = 3000;
        mat.mainTexture = sourceEye.GetCameraTexture();
        eye.InvalidateUploadState();
        UpdateDetectionBuffer(mat, sourceEye.cameraAccess, eye);

        eye.currentMaterial = mat;
        if (eye.meshRenderer != null) eye.meshRenderer.material = mat;
    }

    /// <summary>
    /// Back to the plain passthrough feed.
    /// </summary>
    public void ResetShader()
    {
        ResetEye(leftEye);
        if (rightEye.currentMaterial != null && rightEye.currentMaterial != rightEye.defaultMaterial)
        {
            Destroy(rightEye.currentMaterial);
        }
        ResetEye(rightEye);

        shaderBundleManager.ClearSelection();
        NotifyActiveBundleChanged();
    }

    private static void ResetEye(EyeRenderData eye)
    {
        eye.currentMaterial = eye.defaultMaterial;
        if (eye.meshRenderer != null && eye.defaultMaterial != null) eye.meshRenderer.material = eye.defaultMaterial;
        eye.ReleaseBuffer();
    }

    // ------------------------------------------------------------------ detections

    /// <summary>
    /// Uploads this eye's detections and shared mask atlas to the material only when detections
    /// (or projected right-eye depths) or the bound material actually changed.
    /// </summary>
    private void UpdateDetectionBuffer(Material material, PassthroughCameraAccess eyeCameraAccess, EyeRenderData eyeData)
    {
        bool updated = EnsureEyeBufferData(eyeCameraAccess, eyeData);
        if (updated || eyeData.uploadedMaterial != material)
        {
            BindEyeResourcesToMaterial(material, eyeData);
            eyeData.uploadedMaterial = material;
        }
    }

    private bool EnsureEyeBufferData(PassthroughCameraAccess eyeCameraAccess, EyeRenderData eyeData)
    {
        List<DetectedObject> detections;
        int targetVersion = -1;
        int baseVersion = -1;

        if (detectionTracker != null && eyeCameraAccess != null)
        {
            detections = detectionTracker.GetDetectionsForEye(eyeCameraAccess);
            baseVersion = detectionTracker.Version;
            bool isRightStereo = eyeCameraAccess == rightEye.cameraAccess
                                 && rightEye.cameraAccess != leftEye.cameraAccess;
            targetVersion = isRightStereo ? detectionTracker.RightEyeVersion : baseVersion;
        }
        else
        {
            detections = DetectionProvider?.CurrentDetections ?? NoDetections;
        }

        if (targetVersion >= 0 && eyeData.detectionBuffer != null && eyeData.uploadedVersion == targetVersion)
        {
            return false;
        }

        int detectionCount = detections.Count;
        // Never bind an empty buffer: a zero-sized StructuredBuffer is invalid on some drivers.
        int bufferSize = Mathf.Max(detectionCount, 1);

        if (eyeData.detectionBuffer == null || eyeData.detectionBuffer.count != bufferSize)
        {
            eyeData.ReleaseBuffer();
            eyeData.detectionBuffer = new ComputeBuffer(bufferSize, Marshal.SizeOf<ShaderDetection>());
            eyeData.uploadedMaterial = null;
        }
        if (shaderDetectionsBuffer.Length < bufferSize)
        {
            shaderDetectionsBuffer = new ShaderDetection[bufferSize];
        }

        UpdateSharedMaskAtlas(detections, baseVersion);

        int[] maskIndices = cachedMaskIndices;
        int maskCursor = 0;
        for (int i = 0; i < detectionCount; i++)
        {
            var det = detections[i];
            bool hasMask = det.MaskTexture != null && maskIndices != null && maskCursor < maskIndices.Length;

            shaderDetectionsBuffer[i] = new ShaderDetection
            {
                boundingBox = new Vector4(det.BoundingBox.x, det.BoundingBox.y, det.BoundingBox.width, det.BoundingBox.height),
                confidence = det.Confidence,
                labelID = det.LabelID,
                hasMask = hasMask ? 1 : 0,
                maskIndex = hasMask ? maskIndices[maskCursor++] : -1,
            };
        }

        if (detectionCount == 0)
        {
            shaderDetectionsBuffer[0] = new ShaderDetection { labelID = -1, maskIndex = -1 };
        }

        eyeData.detectionBuffer.SetData(shaderDetectionsBuffer, 0, 0, bufferSize);
        eyeData.uploadedVersion = targetVersion;
        eyeData.uploadedDetectionCount = detectionCount;
        return true;
    }

    private void UpdateSharedMaskAtlas(List<DetectedObject> detections, int baseVersion)
    {
        if (maskAtlas == null)
        {
            cachedMaskIndices = null;
            return;
        }

        if (baseVersion >= 0 && maskAtlasVersion == baseVersion)
        {
            return;
        }

        maskListBuffer.Clear();
        for (int i = 0; i < detections.Count; i++)
        {
            if (detections[i].MaskTexture != null) maskListBuffer.Add(detections[i].MaskTexture);
        }

        if (maskListBuffer.Count > 0)
        {
            cachedMaskIndices = maskAtlas.UpdateMasks(maskListBuffer);
        }
        else
        {
            maskAtlas.UpdateMasks(null);
            cachedMaskIndices = null;
        }

        maskAtlasVersion = baseVersion;
    }

    private void BindEyeResourcesToMaterial(Material material, EyeRenderData eyeData)
    {
        var atlasTexture = cachedMaskIndices != null && cachedMaskIndices.Length > 0 && maskAtlas != null
            ? maskAtlas.AtlasTexture
            : null;

        if (atlasTexture != null)
        {
            material.SetTexture(MaskTextureProperty, atlasTexture);
            material.SetInt(MaskCountProperty, maskAtlas.MaskCount);
            material.SetInt(MaxMasksProperty, maskAtlas.GridSize);
            material.SetInt(HasMaskProperty, 1);
        }
        else
        {
            material.SetInt(HasMaskProperty, 0);
            material.SetInt(MaskCountProperty, 0);
        }

        material.SetInt(DetectionCountProperty, eyeData.uploadedDetectionCount);
        if (eyeData.detectionBuffer != null)
        {
            material.SetBuffer(DetectionsBufferProperty, eyeData.detectionBuffer);
        }
    }
}
