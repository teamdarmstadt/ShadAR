using UnityEngine;

/// <summary>
/// Wires the pipeline together: a finished voice transcription starts a new shader-history
/// session and a shader generation, and every change of the active shader re-configures the
/// detector with that shader's open-vocabulary target list.
/// </summary>
public class GenerativeShaderController : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField] private WhisperTranscriber whisperTranscriber;
    [SerializeField] private LLMShaderGenerator shaderGenerator;
    [Tooltip("A component implementing IDetectionProvider (DetectionNetworkClient).")]
    [SerializeField] private MonoBehaviour detectionProviderBehavior;
    [SerializeField] private ShaderOverlayLoader overlayLoader;
    [SerializeField] private Logger logger;

    private IDetectionProvider DetectionProvider => detectionProviderBehavior as IDetectionProvider;

    private void Awake()
    {
        if (detectionProviderBehavior != null && DetectionProvider == null)
        {
            Debug.LogError($"{nameof(GenerativeShaderController)}: detectionProviderBehavior does not implement IDetectionProvider!", this);
        }
    }

    // Start, not OnEnable: the detection client and the overlay loader initialise in their own
    // Awake/Start, and starting detection before that would hit an unconfigured transport.
    private void Start()
    {
        if (whisperTranscriber != null) whisperTranscriber.OnTranscriptionReady += ProcessPrompt;

        DetectionProvider?.StartDetection();

        if (overlayLoader != null)
        {
            overlayLoader.OnActiveShaderBundleChanged += HandleActiveShaderBundleChanged;
            // Sync the detector with whatever shader is already active.
            HandleActiveShaderBundleChanged(overlayLoader.GetActiveShaderFolder());
        }
    }

    private void OnDestroy()
    {
        if (whisperTranscriber != null) whisperTranscriber.OnTranscriptionReady -= ProcessPrompt;
        if (overlayLoader != null) overlayLoader.OnActiveShaderBundleChanged -= HandleActiveShaderBundleChanged;
        DetectionProvider?.StopDetection();
    }

    private void ProcessPrompt(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            logger?.LogWarning("Empty transcription — nothing was sent to the LLM.", nameof(GenerativeShaderController));
            return;
        }

        logger?.LogInfo($"Processing prompt: {prompt}", nameof(GenerativeShaderController));
        logger?.SetStatus($"Writing shader: \"{prompt}\"", busy: true);

        var session = ShaderHistorySession.StartNewSession();
        session.SaveTranscription(prompt);
        PipelineBenchmarkLogger.Instance?.SetSessionId(session.FolderName);

        _ = shaderGenerator.GenerateShader(prompt, session);
    }

    /// <summary>
    /// Resolves the active shader's open-vocabulary target list and pushes it to the
    /// detector. Falls back to the default vocabulary when the shader has no manifest.
    /// </summary>
    private void HandleActiveShaderBundleChanged(string sessionFolder)
    {
        if (DetectionProvider == null) return;

        if (!string.IsNullOrEmpty(sessionFolder)
            && ShaderDetectionManifest.TryLoad(sessionFolder, out var manifest)
            && manifest.HasClasses)
        {
            DetectionProvider.SetVocabulary(manifest.classes, manifest.confidence, manifest.cocoClasses);
        }
        else
        {
            DetectionProvider.SetVocabulary(null);
        }
    }
}
