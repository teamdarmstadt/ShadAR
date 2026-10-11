using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using OpenAI;
using OpenAI.Chat;
using ChatMessage = OpenAI.Chat.Message;
using OpenAI.Responses;

/// <summary>
/// Generates shader code from transcription text using OpenAI or Claude APIs.
/// Reads a prompt template and combines it with transcription for shader generation.
/// </summary>
public class LLMShaderGenerator : MonoBehaviour
{
    private const string DefaultPromptPath = "Prompts/shader_prompt.txt";
    private const string DefaultObjectDetectionPromptPath = "Prompts/object_detection_prompt.txt";
    private const string ShaderCodeBlockStart = "```shader";
    private const string CodeBlockStart = "```";
    private const string ClaudeApiVersionHeader = "2023-06-01";
    private const string ClaudeFallbackBetaHeader = "server-side-fallback-2026-07-01";
    private static readonly string[] ClaudeEffortLevels = { "low", "medium", "high", "xhigh", "max" };

    // What the menu's effort button cycles through. "" = the provider's default: the parameter is
    // not sent at all, which is the only setting every model accepts (older Claude models and
    // non-reasoning OpenAI models reject an effort with a 400).
    private static readonly string[] ClaudeEffortChoices = { "", "low", "medium", "high", "xhigh", "max" };
    private static readonly string[] OpenAIEffortChoices = { "", "minimal", "low", "medium", "high" };
    // Asks whether the effect is limited to certain things, not whether the user "wants
    // detection": nobody says "detect" in "blur my monitor", and a "no" leaves the LLM without
    // the detection contract, so it guesses COCO ids that never match a slot.
    private const string IntentSystemPrompt =
        "A user describes a visual effect for their live camera view. Reply 'yes' if the effect applies " +
        "only to certain things in the scene (objects, people, animals, body parts, anything that has to be " +
        "found first, e.g. 'blur my monitor', 'make the cat glow', 'pixelate faces'). Reply 'no' if it " +
        "applies to the whole view (e.g. 'black and white', 'add film grain', 'make it look like a cartoon'). " +
        "Reply with only 'yes' or 'no'.";

    // Vocabulary-agnostic intent keywords (EN + DE). Any noun can be a detection target, so
    // these generic action words are only the fast path; the optional intent model check
    // (RunIntentModelCheckAsync) decides everything they do not catch.
    private static readonly string[] ObjectDetectionKeywords =
    {
        "detect",
        "detection",
        "object",
        "bounding box",
        "bbox",
        "mask",
        "segmentation",
        "segment",
        "instance",
        "track",
        "tracking",
        "label",
        "confidence",
        "outline",
        "highlight",
        "silhouette",
        "people",
        "person",
        "mark",
        "marker",
        "every",
        "all ",
        // German equivalents
        "markier",
        "erkenn",
        "hervorheb",
        "umrand",
        "umrahm",
        "alle ",
        "jede",
        "objekt",
        "person",
        "menschen",
        "zeig"
    };

    [Header("Prompt Configuration")]
    [Tooltip("Path to the prompt template file, relative to StreamingAssets")]
    [SerializeField] private string promptFilePath = DefaultPromptPath;

    [Tooltip("Path to the object detection prompt file, relative to StreamingAssets")]
    [SerializeField] private string objectDetectionPromptFilePath = DefaultObjectDetectionPromptPath;

    [Header("Intent Detection")]
    [Tooltip("Use a low-cost LLM check when heuristic matching fails")]
    [SerializeField] private bool enableIntentModelCheck = true;

    [Tooltip("Optional override model for OpenAI intent checks (empty = use main model)")]
    [SerializeField] private string openAIIntentModelOverride = "gpt-5-nano-2025-08-07";

    [Tooltip("Optional override model for Claude intent checks (empty = use main model)")]
    [SerializeField] private string claudeIntentModelOverride = "claude-haiku-4-5";

    [Tooltip("Max tokens for the intent check response")]
    [SerializeField] private int intentMaxTokens = 16;

    [Tooltip("Cache size for intent decisions")]
    [SerializeField] private int intentCacheSize = 128;
    
    [Header("Logging")]
    [Tooltip("Logger for debug output")]
    [SerializeField] private Logger logger;
    
    [Header("Provider Selection")]
    [Tooltip("Default LLM provider for shader generation. Switchable in the headset (settings menu); " +
             "the choice is kept in PlayerPrefs and overrides this value.")]
    [SerializeField] private LlmProvider llmProvider = LlmProvider.OpenAI;
    private const string ProviderPref = "PromptedVision.LlmProvider";

    [Tooltip("Seconds to wait for an LLM answer before giving up with an error. Without a limit, a " +
             "request on a connection that silently died (WiFi power save, NAT dropping a long idle " +
             "request) waits forever and the prompt simply never returns.")]
    [SerializeField] private int requestTimeoutSeconds = 180;

    [Tooltip("Timeout for the short yes/no intent check, which must not hold up the real request.")]
    [SerializeField] private int intentTimeoutSeconds = 15;

    [Tooltip("Claude streams its answer, so a dead connection shows up as silence: abort when no data " +
             "(including the API's keep-alive pings) arrived for this many seconds.")]
    [SerializeField] private int streamIdleTimeoutSeconds = 90;

    [Tooltip("Upper bound for one streamed Claude answer. The most capable models can think for minutes.")]
    [SerializeField] private int streamTimeoutSeconds = 600;

    [Header("OpenAI Configuration")]
    [Tooltip("Reference to the OpenAI Config Manager (optional - will use singleton if not set)")]
    [SerializeField] private OpenAIConfigManager openAIConfigManager;

    [Header("Claude Configuration")]
    [Tooltip("Reference to the Claude Config Manager (optional - will use singleton if not set)")]
    [SerializeField] private ClaudeConfigManager claudeConfigManager;
    
    [Header("Conversation Configuration")]
    [Tooltip("Reference to the Shader Conversation Manager (optional - will use singleton if not set)")]
    [SerializeField] private OpenAIConversationManager conversationManager;

    [Tooltip("Reference to the Claude Conversation Manager (optional - will use singleton if not set)")]
    [SerializeField] private ClaudeConversationManager claudeConversationManager;
    
    [Tooltip("Use conversation mode for context-aware shader generation")]
    [SerializeField] private bool useConversationMode = true;

    [Header("Reasoning Configuration")]
    [Tooltip("Reasoning effort for OpenAI reasoning models. Set to None for non-reasoning models.")]
    [SerializeField] private ReasoningEffortLevel reasoningEffort = ReasoningEffortLevel.None;

    /// <summary>
    /// LLM providers supported by the shader generator.
    /// </summary>
    public enum LlmProvider
    {
        OpenAI,
        Claude
    }

    /// <summary>
    /// Gets the currently selected LLM provider.
    /// </summary>
    public LlmProvider SelectedProvider => Provider;

    /// <summary>
    /// The provider actually used: the pick, except that Claude without an API key falls back to
    /// OpenAI (OpenAI is required anyway for speech-to-text). The pick itself is kept, so adding a
    /// Claude key later brings Claude back.
    /// </summary>
    private LlmProvider Provider =>
        llmProvider == LlmProvider.Claude && !ClaudeAvailable ? LlmProvider.OpenAI : llmProvider;

    /// <summary>Whether a Claude API key is configured; the settings menu greys Claude out otherwise.</summary>
    public bool ClaudeAvailable => (claudeConfigManager ?? ClaudeConfigManager.Instance)?.IsInitialized == true;
    
    /// <summary>
    /// Reasoning effort levels for reasoning models.
    /// </summary>
    public enum ReasoningEffortLevel
    {
        None,   // Don't set reasoning_effort (for non-reasoning models)
        Low,
        Medium,
        High,
        Minimal // last: Unity serializes the index, so inserting it earlier would shift saved values
    }

    private string persistentPromptPath;
    private string persistentObjectDetectionPromptPath;
    private readonly Dictionary<string, bool> intentCache = new Dictionary<string, bool>();
    private readonly Queue<string> intentCacheOrder = new Queue<string>();

    /// <summary>
    /// Event fired when shader code has been generated and saved into the given session.
    /// </summary>
    public event Action<ShaderHistorySession> OnShaderGenerated;

    private void Awake()
    {
        InitializePromptPath();

        int stored = PlayerPrefs.GetInt(ProviderPref, (int)llmProvider);
        if (Enum.IsDefined(typeof(LlmProvider), stored))
        {
            llmProvider = (LlmProvider)stored;
        }
    }

    // Start, not Awake: the config managers load their device config in their own Awake.
    private void Start()
    {
        // Both conversation managers are resolved up front, so switching the provider at runtime
        // finds the other one ready.
        InitializeConversationManager();
        InitializeClaudeConversationManager();
        CheckSelectedProvider();
    }

    /// <summary>
    /// Switches the LLM that writes the shaders and persists the choice. Each provider keeps its
    /// own conversation, so switching back continues where that provider left off.
    /// </summary>
    public void SetProvider(LlmProvider provider)
    {
        if (provider == LlmProvider.Claude && !ClaudeAvailable) return;
        if (llmProvider == provider) return;

        llmProvider = provider;
        PlayerPrefs.SetInt(ProviderPref, (int)provider);
        PlayerPrefs.Save();
        CheckSelectedProvider();
    }

    /// <summary>Model that writes the shaders for the selected provider, or null if it is not configured.</summary>
    public string CurrentShaderModel => Provider == LlmProvider.Claude
        ? (claudeConfigManager ?? ClaudeConfigManager.Instance)?.ShaderModel
        : (openAIConfigManager ?? OpenAIConfigManager.Instance)?.Model;

    /// <summary>Sets the shader model of the selected provider (persisted, wins over the config file).</summary>
    public void SetShaderModel(string model)
    {
        if (string.IsNullOrWhiteSpace(model) || model == CurrentShaderModel) return;

        if (Provider == LlmProvider.Claude)
        {
            (claudeConfigManager ?? ClaudeConfigManager.Instance)?.SetShaderModel(model);
        }
        else
        {
            (openAIConfigManager ?? OpenAIConfigManager.Instance)?.SetModel(model);
        }
        CheckSelectedProvider();
    }

    /// <summary>
    /// Asks the selected provider's API which models this key can use, newest first, so the menu
    /// offers what exists today instead of a list baked into the build. Null when the request
    /// fails or the provider has no key.
    /// </summary>
    public async Task<List<string>> ListShaderModelsAsync()
    {
        try
        {
            var provider = Provider;
            var models = provider == LlmProvider.Claude ? await ListClaudeModelsAsync() : await ListOpenAIModelsAsync();
            if (models == null)
            {
                logger?.LogWarning($"{provider} model list unavailable: provider not configured.", nameof(LLMShaderGenerator));
            }
            else if (models.Count == 0)
            {
                logger?.LogWarning($"{provider} returned no usable shader models.", nameof(LLMShaderGenerator));
            }
            else
            {
                logger?.LogInfo($"{provider} offers {models.Count} shader models (newest: {models[0]}).", nameof(LLMShaderGenerator));
            }
            return models;
        }
        catch (Exception ex)
        {
            logger?.LogWarning($"Could not fetch the {Provider} model list: {ex.Message}", nameof(LLMShaderGenerator));
            return null;
        }
    }

    private async Task<List<string>> ListClaudeModelsAsync()
    {
        var config = claudeConfigManager ?? ClaudeConfigManager.Instance;
        if (config == null || !config.IsInitialized) return null;

        // BaseUrl is .../v1/messages; the model list sits next to it.
        string url = config.BaseUrl.Substring(0, config.BaseUrl.LastIndexOf('/')) + "/models?limit=1000";
        using var request = UnityWebRequest.Get(url);
        request.timeout = intentTimeoutSeconds;
        request.SetRequestHeader("x-api-key", config.ApiKey);
        request.SetRequestHeader("anthropic-version", ClaudeApiVersionHeader);
        await SendWebRequestAsync(request);

        if (request.result != UnityWebRequest.Result.Success)
        {
            throw new Exception($"{request.responseCode} {request.error}");
        }
        // Already sorted newest first by the API.
        return JsonUtility.FromJson<ModelList>(request.downloadHandler.text).data.Select(m => m.id).ToList();
    }

    private async Task<List<string>> ListOpenAIModelsAsync()
    {
        var config = openAIConfigManager ?? OpenAIConfigManager.Instance;
        if (config == null || !config.IsInitialized || config.Client == null) return null;

        var models = await WithTimeout(config.Client.ModelsEndpoint.GetModelsAsync(), intentTimeoutSeconds, "OpenAI model list");
        logger?.LogInfo($"OpenAI lists {models.Count} models in total.", nameof(LLMShaderGenerator));
        return models
            .Where(m => IsOpenAITextModel(m.Id))
            .OrderByDescending(m => m.CreatedAtUnixTimeSeconds)
            .Select(m => m.Id)
            .ToList();
    }

    // ponytail: name-based filter, since /v1/models reports no capabilities. A new non-text
    // family slips through until it gets a token here; a new text family needs no change.
    private static readonly string[] NonTextModelTokens =
        { "audio", "realtime", "tts", "transcribe", "image", "search", "embedding", "instruct", "moderation", "diarize" };

    private static bool IsOpenAITextModel(string id) =>
        (id.StartsWith("gpt-") || (id.Length > 1 && id[0] == 'o' && char.IsDigit(id[1])))
        && !NonTextModelTokens.Any(id.Contains);

    [Serializable] private class ModelList { public ModelEntry[] data; }
    [Serializable] private class ModelEntry { public string id; }

    /// <summary>
    /// Logs which model will write the shaders, or why the selected provider cannot (no API key),
    /// so a switch in the headset gives immediate feedback instead of failing on the next prompt.
    /// </summary>
    private void CheckSelectedProvider()
    {
        switch (Provider)
        {
            case LlmProvider.OpenAI:
                InitializeOpenAIClient();
                var openAI = openAIConfigManager ?? OpenAIConfigManager.Instance;
                if (openAI != null && openAI.IsInitialized)
                {
                    logger?.LogInfo($"Shaders are written by OpenAI ({openAI.Model}).", nameof(LLMShaderGenerator));
                }
                break;
            case LlmProvider.Claude:
                InitializeClaudeConfigManager();
                var claude = claudeConfigManager ?? ClaudeConfigManager.Instance;
                if (claude != null && claude.IsInitialized)
                {
                    logger?.LogInfo($"Shaders are written by Claude ({claude.ShaderModel}).", nameof(LLMShaderGenerator));
                }
                break;
        }
    }

    /// <summary>
    /// Initializes the OpenAI client from the config manager.
    /// </summary>
    private void InitializeOpenAIClient()
    {
        if (Provider != LlmProvider.OpenAI)
        {
            return;
        }

        // Use singleton instance if no direct reference is set
        if (openAIConfigManager == null)
        {
            openAIConfigManager = OpenAIConfigManager.Instance;
        }
        
        if (openAIConfigManager == null)
        {
            logger?.LogError("OpenAIConfigManager is not assigned and singleton instance not found!", nameof(LLMShaderGenerator));
            return;
        }

        if (!openAIConfigManager.IsInitialized)
        {
            logger?.LogError($"OpenAI is not configured — add an API key to {openAIConfigManager.GetDeviceConfigPath()}.", nameof(LLMShaderGenerator));
            return;
        }

        // Verify client is available
        if (openAIConfigManager.Client == null)
        {
            logger?.LogError("Failed to get OpenAI client from OpenAIConfigManager!", nameof(LLMShaderGenerator));
        }
    }

    /// <summary>
    /// Initializes the Claude config manager.
    /// </summary>
    private void InitializeClaudeConfigManager()
    {
        if (Provider != LlmProvider.Claude)
        {
            return;
        }

        if (claudeConfigManager == null)
        {
            claudeConfigManager = ClaudeConfigManager.Instance;
        }

        if (claudeConfigManager == null)
        {
            logger?.LogError("ClaudeConfigManager is not assigned and singleton instance not found!", nameof(LLMShaderGenerator));
            return;
        }

        if (!claudeConfigManager.IsInitialized)
        {
            logger?.LogError($"Claude is not configured — add an API key to {claudeConfigManager.GetDeviceConfigPath()}.", nameof(LLMShaderGenerator));
        }
    }

    /// <summary>
    /// Initializes the Claude conversation manager.
    /// </summary>
    private void InitializeClaudeConversationManager()
    {
        if (!useConversationMode)
        {
            logger?.LogDebug("Conversation mode is disabled.", nameof(LLMShaderGenerator));
            return;
        }

        if (claudeConversationManager == null)
        {
            claudeConversationManager = ClaudeConversationManager.Instance;
        }

        if (claudeConversationManager == null)
        {
            logger?.LogWarning("ClaudeConversationManager not found. Conversation mode will be disabled.", nameof(LLMShaderGenerator));
            logger?.LogWarning("Add ClaudeConversationManager component to the scene to enable conversation mode.", nameof(LLMShaderGenerator));
            useConversationMode = false;
        }
        else
        {
            logger?.LogDebug($"Claude conversation mode enabled. {claudeConversationManager.GetConversationSummary()}", nameof(LLMShaderGenerator));
        }
    }
    
    /// <summary>
    /// Initializes the conversation manager.
    /// </summary>
    private void InitializeConversationManager()
    {
        if (!useConversationMode)
        {
            logger?.LogDebug("Conversation mode is disabled.", nameof(LLMShaderGenerator));
            return;
        }
        
        // Use singleton instance if no direct reference is set
        if (conversationManager == null)
        {
            conversationManager = OpenAIConversationManager.Instance;
        }
        
        if (conversationManager == null)
        {
            logger?.LogWarning("OpenAIConversationManager not found. Conversation mode will be disabled.", nameof(LLMShaderGenerator));
            logger?.LogWarning("Add OpenAIConversationManager component to the scene to enable conversation mode.", nameof(LLMShaderGenerator));
            useConversationMode = false;
        }
        else
        {
            logger?.LogDebug($"OpenAI conversation mode enabled. {conversationManager.GetConversationSummary()}", nameof(LLMShaderGenerator));
        }
    }
    
    /// <summary>
    /// Initializes prompt path and copies template from StreamingAssets if needed.
    /// </summary>
    private void InitializePromptPath()
    {
        persistentPromptPath = Path.Combine(Application.persistentDataPath, promptFilePath);
        persistentObjectDetectionPromptPath = Path.Combine(Application.persistentDataPath, objectDetectionPromptFilePath);

#if UNITY_ANDROID && !UNITY_EDITOR
        EnsurePromptCopiedAndroid(promptFilePath, persistentPromptPath);
        EnsurePromptCopiedAndroid(objectDetectionPromptFilePath, persistentObjectDetectionPromptPath);
#else
        EnsurePromptCopied(promptFilePath, persistentPromptPath);
        EnsurePromptCopied(objectDetectionPromptFilePath, persistentObjectDetectionPromptPath);
#endif
    }

    /// <summary>
    /// Sidecar holding the hash of the SHIPPED prompt the on-device copy was made from.
    /// This is what makes "did the app ship a new prompt?" answerable without comparing
    /// against the on-device file itself — which the user is allowed to edit.
    /// </summary>
    private static string ShippedHashPath(string persistentPath) => persistentPath + ".shipped";

    /// <summary>
    /// Decides whether the on-device prompt has to be (re)written, and does it.
    ///
    /// The old rule was "copy only if absent", which silently served a stale prompt forever:
    /// updating a prompt in StreamingAssets had no effect on any headset that had run the app
    /// before, and the only symptom was the LLM ignoring instructions it never received.
    ///
    /// The rule is now "copy when the SHIPPED prompt differs from the one this copy was made
    /// from". A hand-edited on-device prompt is therefore preserved (editing it does not change
    /// the shipped hash), while a genuine app-side prompt change propagates on next launch.
    /// An install predating the sidecar has no recorded hash and is refreshed once.
    /// </summary>
    private void WritePromptIfShippedVersionChanged(string shippedText, string persistentPath)
    {
        if (string.IsNullOrEmpty(shippedText))
        {
            return;
        }

        string shippedHash = ComputeHash(shippedText);
        string hashPath = ShippedHashPath(persistentPath);
        string recordedHash = File.Exists(hashPath) ? File.ReadAllText(hashPath).Trim() : null;

        if (File.Exists(persistentPath) && recordedHash == shippedHash)
        {
            return;
        }

        bool isUpdate = File.Exists(persistentPath);
        Directory.CreateDirectory(Path.GetDirectoryName(persistentPath));
        File.WriteAllText(persistentPath, shippedText);
        File.WriteAllText(hashPath, shippedHash);

        logger?.LogInfo(
            isUpdate
                ? $"Prompt updated from the app's shipped version: {Path.GetFileName(persistentPath)}"
                : $"Prompt installed: {Path.GetFileName(persistentPath)}",
            nameof(LLMShaderGenerator));
    }

    private static string ComputeHash(string text)
    {
        using var md5 = System.Security.Cryptography.MD5.Create();
        return System.Convert.ToBase64String(md5.ComputeHash(System.Text.Encoding.UTF8.GetBytes(text)));
    }

    private void EnsurePromptCopied(string promptPath, string persistentPath)
    {
        string sourcePath = Path.Combine(Application.streamingAssetsPath, promptPath);
        if (!File.Exists(sourcePath))
        {
            logger?.LogWarning($"Shipped prompt missing: {sourcePath}", nameof(LLMShaderGenerator));
            return;
        }

        WritePromptIfShippedVersionChanged(File.ReadAllText(sourcePath), persistentPath);
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    private void EnsurePromptCopiedAndroid(string promptPath, string persistentPath)
    {
        // Unlike the desktop path this cannot short-circuit on File.Exists: inside the APK the
        // shipped prompt is only readable through UnityWebRequest, so it must be fetched before
        // its hash can be compared. It is a few KB once per launch.
        string sourcePath = Path.Combine(Application.streamingAssetsPath, promptPath);
        StartCoroutine(CopyPromptFromStreamingAssetsAndroid(sourcePath, persistentPath));
    }
#endif

#if UNITY_ANDROID && !UNITY_EDITOR
    /// <summary>
    /// Copies prompt file from StreamingAssets on Android using UnityWebRequest.
    /// </summary>
    private System.Collections.IEnumerator CopyPromptFromStreamingAssetsAndroid(string sourcePath, string destPath)
    {
        string uri = sourcePath;
        if (!uri.StartsWith("jar:") && !uri.StartsWith("file://"))
        {
            uri = "file://" + uri;
        }
        
        using (var www = UnityEngine.Networking.UnityWebRequest.Get(uri))
        {
            yield return www.SendWebRequest();
            
            if (www.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                WritePromptIfShippedVersionChanged(www.downloadHandler.text, destPath);
            }
            else
            {
                logger?.LogError($"Failed to copy prompt file on Android: {uri}", nameof(LLMShaderGenerator));
            }
        }
    }
#endif

    /// <summary>
    /// Generates shader code from transcription using the selected provider and stores it in
    /// <paramref name="session"/>. The session is passed in rather than looked up, because a
    /// second prompt may start a new session while this one is still waiting for the LLM.
    /// </summary>
    public async Task GenerateShader(string transcription, ShaderHistorySession session)
    {
        string providerName = Provider.ToString();

        try
        {
            string shaderCode = null;

            switch (Provider)
            {
                case LlmProvider.OpenAI:
                    shaderCode = await GenerateWithOpenAI(transcription);
                    break;
                case LlmProvider.Claude:
                    shaderCode = await GenerateWithClaudeAPI(transcription);
                    break;
            }
            
            if (!string.IsNullOrEmpty(shaderCode))
            {
                session.SaveShaderCode(shaderCode);
                logger?.LogInfo($"Shader code saved to: {session.ShaderFilePath}", nameof(LLMShaderGenerator));

                // Persist the open-vocabulary target list next to the shader so the
                // detector can be configured with exactly this shader's classes when
                // it is activated (slot index = shader labelID).
                if (TryParseDetectionManifest(shaderCode, out var manifest))
                {
                    session.SaveDetectionClasses(manifest);
                    logger?.LogInfo($"Detection classes saved: [{string.Join(", ", manifest.classes)}]", nameof(LLMShaderGenerator));

                    // Loud, because the symptom is otherwise invisible: the shader works fine
                    // on the open-vocabulary detector and only misbehaves after switching to
                    // the COCO one, where the server then has to guess the translation. The
                    // usual cause is a stale on-device prompt that never carried the rule.
                    if (manifest.cocoClasses == null || manifest.cocoClasses.Count != manifest.classes.Count)
                    {
                        logger?.LogWarning(
                            "Shader has no usable @detect-coco line — the COCO detector will fall back to " +
                            "name matching for it. If this is a freshly generated shader, the prompt the LLM " +
                            "saw is out of date.", nameof(LLMShaderGenerator));
                    }
                }

                OnShaderGenerated?.Invoke(session);
            }
            else
            {
                logger?.LogError($"No shader code received from {providerName}.", nameof(LLMShaderGenerator));
            }
        }
        catch (TimeoutException ex)
        {
            logger?.LogError($"{providerName}: {ex.Message} Please try the prompt again.", nameof(LLMShaderGenerator));
        }
        catch (Exception ex)
        {
            logger?.LogError($"Shader generation failed: {ex}", nameof(LLMShaderGenerator));
        }
    }

    /// <summary>
    /// Waits for an OpenAI SDK request with a deadline. The SDK only honours cancellation on
    /// streaming calls, so a request hanging on a dead connection cannot be aborted — we stop
    /// waiting and report it instead, and the orphaned request is discarded whenever it ends.
    /// </summary>
    private static async Task<T> WithTimeout<T>(Task<T> request, int seconds, string what)
    {
        if (await Task.WhenAny(request, Task.Delay(TimeSpan.FromSeconds(seconds))) != request)
        {
            throw new TimeoutException($"{what} did not answer within {seconds} s.");
        }
        return await request;
    }

    /// <summary>
    /// Generates shader using OpenAI APIs.
    /// </summary>
    private async Task<string> GenerateWithOpenAI(string transcription)
    {
        // Get fresh reference from singleton each time
        var configManager = openAIConfigManager ?? OpenAIConfigManager.Instance;
        
        if (configManager == null || !configManager.IsInitialized)
        {
            logger?.LogError("OpenAIConfigManager is not available or not initialized!", nameof(LLMShaderGenerator));
            return null;
        }
        
        var client = configManager.Client;
        
        if (client == null)
        {
            logger?.LogError("OpenAIClient is not initialized!", nameof(LLMShaderGenerator));
            return null;
        }

        string model = configManager.Model;

        PipelineBenchmarkLogger.Instance?.RecordTimestamp("sent_to_openai");
        PipelineBenchmarkLogger.Instance?.SetReasoningEffort(
            configManager.ReasoningEffort ?? reasoningEffort.ToString().ToLowerInvariant());

        bool includeObjectDetectionPrompt = await ShouldIncludeObjectDetectionPromptAsync(transcription);

        // Try to use Conversations API if enabled
        if (!includeObjectDetectionPrompt && useConversationMode && conversationManager != null && conversationManager.IsConversationModeActive)
        {
            return await GenerateWithConversationAPI(client, transcription, model);
        }

        if (includeObjectDetectionPrompt && useConversationMode && conversationManager != null && conversationManager.IsConversationModeActive)
        {
            logger?.LogDebug("Object detection prompt requested. Using Chat API to apply expanded system prompt.", nameof(LLMShaderGenerator));
        }

        // Fallback to Chat API
        return await GenerateWithChatAPI(client, transcription, model, includeObjectDetectionPrompt);
    }

    /// <summary>
    /// Generates shader using the Conversations API (maintains context).
    /// </summary>
    private async Task<string> GenerateWithConversationAPI(OpenAIClient client, string transcription, string model)
    {
        logger?.LogInfo($"Using Conversations API with model: {model}", nameof(LLMShaderGenerator));

        var conversation = conversationManager.GetConversation();

        if (conversation == null)
        {
            logger?.LogError("Conversation is null. Falling back to Chat API.", nameof(LLMShaderGenerator));
            return await GenerateWithChatAPI(client, transcription, model);
        }

        string conversationId = conversationManager.GetConversationId();

        try
        {
            // The Responses API appends this input (and the answer) to the conversation itself;
            // adding the user message separately first would store every prompt twice.
            var request = new CreateResponseRequest(
                textInput: transcription,
                conversationId: conversationId,
                model: model,
                reasoning: ToReasoning(GetReasoningEffort())
            );

            var response = await WithTimeout(
                client.ResponsesEndpoint.CreateModelResponseAsync(request), requestTimeoutSeconds, "OpenAI");

            if (response?.Output != null)
            {
                var responseItem = response.Output.LastOrDefault();
                
                if (responseItem != null)
                {
                    logger?.LogDebug($"Received response from conversation {conversationId}", nameof(LLMShaderGenerator));
                    
                    // Update message count
                    conversationManager.IncrementMessageCount();
                    
                    string content = responseItem.ToString();
                    return ExtractShaderCode(content);
                }
            }
        }
        catch (Exception ex) when (ex is not TimeoutException)
        {
            logger?.LogWarning($"Conversations API error: {ex.Message}. Falling back to Chat API.", nameof(LLMShaderGenerator));
            return await GenerateWithChatAPI(client, transcription, model);
        }

        logger?.LogWarning("No response from Conversations API. Falling back to Chat API.", nameof(LLMShaderGenerator));
        return await GenerateWithChatAPI(client, transcription, model);
    }

    /// <summary>
    /// Generates shader using the Chat API (single-shot, no context).
    /// </summary>
    private async Task<string> GenerateWithChatAPI(
        OpenAIClient client,
        string transcription,
        string model,
        bool? includeObjectDetectionPrompt = null)
    {
        logger?.LogInfo($"Using Chat API with model: {model}", nameof(LLMShaderGenerator));

        bool includeDetection = includeObjectDetectionPrompt ?? await ShouldIncludeObjectDetectionPromptAsync(transcription);
        string systemPrompt = BuildSystemPrompt(includeDetection);
        if (string.IsNullOrEmpty(systemPrompt))
        {
            logger?.LogError("System prompt is empty or missing.", nameof(LLMShaderGenerator));
            return null;
        }

        var chatRequest = new ChatRequest(
            new[]
            {
                new ChatMessage(Role.System, systemPrompt),
                new ChatMessage(Role.User, transcription)
            },
            model: model,
            reasoningEffort: GetReasoningEffort()
        );

        var chatResponse = await WithTimeout(
            client.ChatEndpoint.GetCompletionAsync(chatRequest), requestTimeoutSeconds, "OpenAI");

        var choice = chatResponse?.FirstChoice;
        if (choice == null)
        {
            return null;
        }

        if (choice.FinishReason == "length")
        {
            logger?.LogError("OpenAI stopped at the output token limit — the shader is cut off.", nameof(LLMShaderGenerator));
            return null;
        }

        return ExtractShaderCode((string)choice.Message.Content);
    }

    /// <summary>
    /// Generates shader using the Claude Messages API.
    /// </summary>
    private async Task<string> GenerateWithClaudeAPI(string transcription)
    {
        var configManager = claudeConfigManager ?? ClaudeConfigManager.Instance;
        if (configManager == null || !configManager.IsInitialized)
        {
            logger?.LogError("ClaudeConfigManager is not available or not initialized!", nameof(LLMShaderGenerator));
            return null;
        }

        logger?.LogInfo($"Using Claude API with model: {configManager.ShaderModel}", nameof(LLMShaderGenerator));

        bool includeObjectDetectionPrompt = await ShouldIncludeObjectDetectionPromptAsync(transcription);
        string baseSystemPrompt = BuildSystemPrompt(includeObjectDetectionPrompt);
        if (string.IsNullOrEmpty(baseSystemPrompt))
        {
            logger?.LogError("System prompt is empty or missing.", nameof(LLMShaderGenerator));
            return null;
        }

        string systemPrompt = baseSystemPrompt;
        bool conversation = useConversationMode && claudeConversationManager != null && claudeConversationManager.IsConversationModeActive;
        var messages = new List<ClaudeMessage>();

        if (conversation)
        {
            systemPrompt = await claudeConversationManager.GetSystemPromptAsync();
            if (string.IsNullOrWhiteSpace(systemPrompt))
            {
                logger?.LogWarning("Claude system prompt missing. Falling back to prompt template.", nameof(LLMShaderGenerator));
                systemPrompt = baseSystemPrompt;
            }

            if (includeObjectDetectionPrompt)
            {
                string detectionPrompt = ReadObjectDetectionPrompt();
                if (!string.IsNullOrWhiteSpace(detectionPrompt))
                {
                    systemPrompt = $"{systemPrompt}\n\n{detectionPrompt}";
                }
            }

            messages = BuildClaudeMessagesFromHistory(claudeConversationManager.GetMessagesSnapshot());
        }
        messages.Add(BuildClaudeUserMessage(transcription));

        PipelineBenchmarkLogger.Instance?.RecordTimestamp("sent_to_claude");

        var requestPayload = new ClaudeRequestBase
        {
            model = configManager.ShaderModel,
            max_tokens = configManager.ShaderMaxTokens,
            system = systemPrompt,
            messages = messages.ToArray()
        };

        ClaudeStreamResult result = await StreamClaudeRequestAsync(BuildShaderRequestJson(requestPayload, configManager), configManager);

        if (result.stalled)
        {
            logger?.LogError($"Claude stream stalled: no data for {streamIdleTimeoutSeconds} s. Please try the prompt again.", nameof(LLMShaderGenerator));
            return null;
        }

        if (!result.success)
        {
            logger?.LogError($"Claude request failed ({(int)result.statusCode}): {result.error}", nameof(LLMShaderGenerator));
            return null;
        }

        if (!string.IsNullOrEmpty(result.servedModel) && result.servedModel != configManager.ShaderModel)
        {
            logger?.LogInfo($"{configManager.ShaderModel} declined; served by fallback model {result.servedModel}.", nameof(LLMShaderGenerator));
        }

        switch (result.stopReason)
        {
            // A refusal can arrive before any output or mid-stream; either way the text is not a
            // usable answer. With server-side fallbacks on, it means every model in the chain declined.
            case "refusal":
                logger?.LogError($"Claude declined the request (category: {result.refusalCategory ?? "none"}). " +
                                 "Rephrase the prompt or pick another model.", nameof(LLMShaderGenerator));
                return null;
            case "max_tokens":
                logger?.LogError($"Claude stopped at max_tokens ({configManager.ShaderMaxTokens}) — the shader is cut off. " +
                                 "Raise maxTokens in claude_config.json.", nameof(LLMShaderGenerator));
                return null;
        }

        string content = result.text;
        if (string.IsNullOrWhiteSpace(content))
        {
            logger?.LogError($"Claude response had no text (stop_reason: {result.stopReason ?? "none"}).", nameof(LLMShaderGenerator));
            return null;
        }

        // Only a completed exchange enters the history, so a failed request never leaves a
        // dangling user turn behind.
        if (conversation)
        {
            claudeConversationManager.AddUserMessage(transcription);
            claudeConversationManager.AddAssistantMessage(content);
        }

        return ExtractShaderCode(content);
    }

    private async Task<ClaudeRequestResult> SendClaudeRequestAsync(string json, ClaudeConfigManager configManager, int timeoutSeconds)
    {
        using var request = new UnityWebRequest(configManager.BaseUrl, UnityWebRequest.kHttpVerbPOST);
        request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
        request.downloadHandler = new DownloadHandlerBuffer();
        request.timeout = timeoutSeconds;
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("x-api-key", configManager.ApiKey);
        request.SetRequestHeader("anthropic-version", ClaudeApiVersionHeader);

        await SendWebRequestAsync(request);

        var result = new ClaudeRequestResult
        {
            success = request.result == UnityWebRequest.Result.Success,
            statusCode = request.responseCode,
            raw = request.downloadHandler?.text,
            transportError = request.error
        };

        if (!string.IsNullOrWhiteSpace(result.raw))
        {
            try
            {
                result.response = JsonUtility.FromJson<ClaudeResponse>(result.raw);
            }
            catch
            {
                // Ignore parse errors; raw response will be logged by caller.
            }
        }

        return result;
    }

    /// <summary>
    /// Adds the fields JsonUtility cannot leave out: streaming, the effort level and the
    /// server-side refusal fallback ("default" lets Anthropic pick the fallback model per
    /// refusal category).
    /// </summary>
    private string BuildShaderRequestJson(ClaudeRequestBase payload, ClaudeConfigManager configManager)
    {
        var extra = new StringBuilder(",\"stream\":true");

        string effort = configManager.ShaderEffort;
        if (Array.IndexOf(ClaudeEffortLevels, effort) >= 0)
        {
            extra.Append(",\"output_config\":{\"effort\":\"").Append(effort).Append("\"}");
        }
        else if (!string.IsNullOrEmpty(effort))
        {
            logger?.LogWarning($"Ignoring unknown Claude effort '{effort}'.", nameof(LLMShaderGenerator));
        }

        if (configManager.UseServerSideFallbacks)
        {
            extra.Append(",\"fallbacks\":\"default\"");
        }

        string json = JsonUtility.ToJson(payload);
        return json.Substring(0, json.Length - 1) + extra + "}";
    }

    /// <summary>
    /// Sends a streaming Messages API request and collects the answer. Streaming keeps bytes
    /// flowing while the model thinks, so a long generation no longer looks like a dead
    /// connection to WiFi power saving or NAT, and a connection that does die is detected by
    /// silence instead of waiting for the overall timeout.
    /// </summary>
    private async Task<ClaudeStreamResult> StreamClaudeRequestAsync(string json, ClaudeConfigManager configManager)
    {
        var handler = new ClaudeStreamHandler();
        using var request = new UnityWebRequest(configManager.BaseUrl, UnityWebRequest.kHttpVerbPOST);
        request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
        request.downloadHandler = handler;
        request.timeout = streamTimeoutSeconds;
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("x-api-key", configManager.ApiKey);
        request.SetRequestHeader("anthropic-version", ClaudeApiVersionHeader);
        if (configManager.UseServerSideFallbacks)
        {
            request.SetRequestHeader("anthropic-beta", ClaudeFallbackBetaHeader);
        }

        var operation = request.SendWebRequest();
        bool stalled = false;
        while (!operation.isDone)
        {
            if (Time.realtimeSinceStartup - handler.LastDataTime > streamIdleTimeoutSeconds)
            {
                request.Abort();
                stalled = true;
                break;
            }
            await Task.Yield();
        }

        bool httpOk = request.result == UnityWebRequest.Result.Success;
        return new ClaudeStreamResult
        {
            success = !stalled && httpOk && handler.ErrorMessage == null,
            stalled = stalled,
            statusCode = request.responseCode,
            // A non-2xx answer is a plain JSON error body, not an event stream.
            error = handler.ErrorMessage ?? (httpOk ? null : $"{request.error}; response={handler.RawText}"),
            text = handler.Text.ToString(),
            stopReason = handler.StopReason,
            refusalCategory = handler.RefusalCategory,
            servedModel = handler.ServedModel,
        };
    }

    /// <summary>
    /// Parses the server-sent events of a streamed Messages API response as they arrive.
    /// Unity calls ReceiveData on the main thread.
    /// </summary>
    private sealed class ClaudeStreamHandler : DownloadHandlerScript
    {
        private const int MaxRawChars = 16 * 1024;

        private readonly Decoder decoder = Encoding.UTF8.GetDecoder();
        private readonly StringBuilder line = new();
        private readonly StringBuilder raw = new();
        private char[] chars = new char[16 * 1024];

        public readonly StringBuilder Text = new();
        public string StopReason { get; private set; }
        public string RefusalCategory { get; private set; }
        public string ServedModel { get; private set; }
        public string ErrorMessage { get; private set; }
        public float LastDataTime { get; private set; } = Time.realtimeSinceStartup;
        public string RawText => raw.ToString();

        public ClaudeStreamHandler() : base(new byte[16 * 1024]) { }

        protected override bool ReceiveData(byte[] data, int dataLength)
        {
            LastDataTime = Time.realtimeSinceStartup;

            // A UTF-8 character can be split across two chunks; the Decoder carries the rest over.
            int needed = decoder.GetCharCount(data, 0, dataLength);
            if (chars.Length < needed) chars = new char[needed];
            int count = decoder.GetChars(data, 0, dataLength, chars, 0);

            if (raw.Length < MaxRawChars) raw.Append(chars, 0, Math.Min(count, MaxRawChars - raw.Length));

            for (int i = 0; i < count; i++)
            {
                char c = chars[i];
                if (c == '\n')
                {
                    ProcessLine(line.ToString());
                    line.Clear();
                }
                else if (c != '\r')
                {
                    line.Append(c);
                }
            }
            return true;
        }

        private void ProcessLine(string text)
        {
            if (!text.StartsWith("data:")) return;

            var evt = JsonUtility.FromJson<ClaudeStreamEvent>(text.Substring(5));
            switch (evt?.type)
            {
                case "message_start":
                    // After a server-side fallback this already names the fallback model.
                    ServedModel = evt.message?.model;
                    break;
                case "content_block_delta":
                    // Text before and after a mid-stream fallback block belongs together: the
                    // fallback model continues the partial answer. Thinking deltas are ignored.
                    if (evt.delta?.type == "text_delta") Text.Append(evt.delta.text);
                    break;
                case "message_delta":
                    if (!string.IsNullOrEmpty(evt.delta?.stop_reason))
                    {
                        StopReason = evt.delta.stop_reason;
                        string category = evt.delta.stop_details?.category;
                        RefusalCategory = string.IsNullOrEmpty(category) ? null : category;
                    }
                    break;
                case "error":
                    ErrorMessage = $"{evt.error?.type}: {evt.error?.message}";
                    break;
            }
        }
    }

    /// <summary>
    /// Runs a UnityWebRequest asynchronously.
    /// </summary>
    private static Task<UnityWebRequest> SendWebRequestAsync(UnityWebRequest request)
    {
        var tcs = new TaskCompletionSource<UnityWebRequest>();
        var operation = request.SendWebRequest();
        operation.completed += _ => tcs.TrySetResult(request);
        return tcs.Task;
    }

    /// <summary>
    /// Extracts the first text block from a Claude response.
    /// </summary>
    private static string ExtractClaudeText(ClaudeResponse response)
    {
        if (response?.content == null || response.content.Length == 0)
        {
            return null;
        }

        foreach (var block in response.content)
        {
            if (block != null && string.Equals(block.type, "text", StringComparison.OrdinalIgnoreCase))
            {
                return block.text;
            }
        }

        return response.content[0]?.text;
    }


    private static ClaudeMessage BuildClaudeUserMessage(string text)
    {
        return new ClaudeMessage
        {
            role = "user",
            content = new[] { new ClaudeContent { type = "text", text = text } }
        };
    }

    private static List<ClaudeMessage> BuildClaudeMessagesFromHistory(IReadOnlyList<ClaudeConversationManager.ConversationMessage> history)
    {
        var messages = new List<ClaudeMessage>();
        if (history == null)
        {
            return messages;
        }

        foreach (var item in history)
        {
            if (item == null || string.IsNullOrWhiteSpace(item.content))
            {
                continue;
            }

            messages.Add(new ClaudeMessage
            {
                role = string.IsNullOrWhiteSpace(item.role) ? "user" : item.role,
                content = new[] { new ClaudeContent { type = "text", text = item.content } }
            });
        }

        return messages;
    }

    /// <summary>
    /// Extracts shader code from GPT response, removing markdown code blocks.
    /// </summary>
    private string ExtractShaderCode(string response)
    {
        int startIdx = response.IndexOf(ShaderCodeBlockStart);
        if (startIdx >= 0)
        {
            startIdx += ShaderCodeBlockStart.Length;
        }
        else
        {
            startIdx = response.IndexOf(CodeBlockStart);
            if (startIdx >= 0)
            {
                startIdx += CodeBlockStart.Length;
            }
        }
        
        if (startIdx >= 0)
        {
            int endIdx = response.IndexOf(CodeBlockStart, startIdx);
            if (endIdx > startIdx)
            {
                return response[startIdx..endIdx].Trim('\r', '\n');
            }
        }
        
        return response.Trim();
    }

    /// <summary>
    /// Gets the reasoning effort parameter for the API request.
    /// Prefers the value from OpenAIConfigManager (device config), falls back to local Inspector setting.
    /// Returns null if reasoning effort is set to None.
    /// </summary>
    private OpenAI.ReasoningEffort? GetReasoningEffort()
    {
        // Resolve effective level: config manager (device override) takes priority over Inspector
        var effectiveLevel = reasoningEffort;
        
        var configManager = openAIConfigManager ?? OpenAIConfigManager.Instance;
        if (configManager != null && !string.IsNullOrWhiteSpace(configManager.ReasoningEffort))
        {
            if (Enum.TryParse<ReasoningEffortLevel>(configManager.ReasoningEffort, ignoreCase: true, out var parsed))
            {
                effectiveLevel = parsed;
            }
            else
            {
                logger?.LogWarning($"Invalid reasoning effort in config: '{configManager.ReasoningEffort}'. Using Inspector value: {reasoningEffort}", nameof(LLMShaderGenerator));
            }
        }
        
        return effectiveLevel switch
        {
            ReasoningEffortLevel.None => null,
            ReasoningEffortLevel.Minimal => OpenAI.ReasoningEffort.Minimal,
            ReasoningEffortLevel.Low => OpenAI.ReasoningEffort.Low,
            ReasoningEffortLevel.Medium => OpenAI.ReasoningEffort.Medium,
            ReasoningEffortLevel.High => OpenAI.ReasoningEffort.High,
            _ => null
        };
    }
    
    // ponytail: built through JSON because the SDK's public constructor also requests reasoning
    // summaries (summary=auto), which organisations without verification get rejected for. The
    // JSON constructor leaves the summary out.
    private static Reasoning ToReasoning(OpenAI.ReasoningEffort? effort) => effort == null
        ? null
        : Newtonsoft.Json.JsonConvert.DeserializeObject<Reasoning>($"{{\"effort\":\"{effort.Value.ToString().ToLowerInvariant()}\"}}");

    /// <summary>Effort levels the menu offers for the selected provider; "" = provider default.</summary>
    public IReadOnlyList<string> ShaderEffortChoices =>
        Provider == LlmProvider.Claude ? ClaudeEffortChoices : OpenAIEffortChoices;

    /// <summary>Effort the selected provider currently uses; "" = provider default.</summary>
    public string CurrentShaderEffort
    {
        get
        {
            if (Provider == LlmProvider.Claude)
            {
                return (claudeConfigManager ?? ClaudeConfigManager.Instance)?.ShaderEffort ?? "";
            }
            string openAI = (openAIConfigManager ?? OpenAIConfigManager.Instance)?.ReasoningEffort ?? "";
            return openAI.Equals("None", StringComparison.OrdinalIgnoreCase) ? "" : openAI.ToLowerInvariant();
        }
    }

    /// <summary>Sets the selected provider's shader effort (persisted, wins over the config file).</summary>
    public void SetShaderEffort(string effort)
    {
        if (Provider == LlmProvider.Claude)
        {
            (claudeConfigManager ?? ClaudeConfigManager.Instance)?.SetShaderEffort(effort);
        }
        else
        {
            (openAIConfigManager ?? OpenAIConfigManager.Instance)?.SetReasoningEffort(string.IsNullOrEmpty(effort) ? "None" : effort);
        }
        logger?.LogInfo($"{Provider} shader effort: {(string.IsNullOrEmpty(effort) ? "model default" : effort)}.", nameof(LLMShaderGenerator));
    }

    private string ReadPromptTemplate()
    {
        return ReadPromptFile(persistentPromptPath, "Prompt template");
    }

    private string ReadObjectDetectionPrompt()
    {
        return ReadPromptFile(persistentObjectDetectionPromptPath, "Object detection prompt");
    }

    private string ReadPromptFile(string persistentPath, string label)
    {
        if (string.IsNullOrWhiteSpace(persistentPath))
        {
            logger?.LogError($"{label} path is not set.", nameof(LLMShaderGenerator));
            return null;
        }

        if (!File.Exists(persistentPath))
        {
            logger?.LogError($"{label} not found in persistent storage: {persistentPath}", nameof(LLMShaderGenerator));
            return null;
        }

        string content = File.ReadAllText(persistentPath);
        if (string.IsNullOrWhiteSpace(content))
        {
            logger?.LogError($"{label} is empty: {persistentPath}", nameof(LLMShaderGenerator));
            return null;
        }

        return content;
    }

    private string BuildSystemPrompt(bool includeObjectDetectionPrompt)
    {
        string basePrompt = ReadPromptTemplate();
        if (string.IsNullOrWhiteSpace(basePrompt))
        {
            return null;
        }

        if (!includeObjectDetectionPrompt)
        {
            return basePrompt;
        }

        string detectionPrompt = ReadObjectDetectionPrompt();
        if (string.IsNullOrWhiteSpace(detectionPrompt))
        {
            logger?.LogWarning("Object detection prompt missing; continuing with base system prompt.", nameof(LLMShaderGenerator));
            return basePrompt;
        }

        return $"{basePrompt}\n\n{detectionPrompt}";
    }

    private async Task<bool> ShouldIncludeObjectDetectionPromptAsync(string transcription)
    {
        if (ShouldIncludeObjectDetectionPromptHeuristic(transcription))
        {
            CacheIntentDecision(transcription, true);
            logger?.LogInfo("Object detection: yes (keyword)", nameof(LLMShaderGenerator));
            return true;
        }

        if (!enableIntentModelCheck)
        {
            return false;
        }

        if (TryGetCachedIntentDecision(transcription, out bool cachedDecision))
        {
            return cachedDecision;
        }

        bool decision = await RunIntentModelCheckAsync(transcription);
        CacheIntentDecision(transcription, decision);
        logger?.LogInfo($"Object detection: {(decision ? "yes" : "no")} (intent model)", nameof(LLMShaderGenerator));
        return decision;
    }

    private bool ShouldIncludeObjectDetectionPromptHeuristic(string transcription)
    {
        if (string.IsNullOrWhiteSpace(transcription))
        {
            return false;
        }

        if (ContainsAnyKeyword(transcription, ObjectDetectionKeywords))
        {
            return true;
        }

        return false;
    }

    private async Task<bool> RunIntentModelCheckAsync(string transcription)
    {
        if (string.IsNullOrWhiteSpace(transcription))
        {
            return false;
        }

        switch (Provider)
        {
            case LlmProvider.OpenAI:
                return await RunOpenAIIntentCheckAsync(transcription);
            case LlmProvider.Claude:
                return await RunClaudeIntentCheckAsync(transcription);
            default:
                return false;
        }
    }

    private async Task<bool> RunOpenAIIntentCheckAsync(string transcription)
    {
        var configManager = openAIConfigManager ?? OpenAIConfigManager.Instance;
        if (configManager == null || !configManager.IsInitialized)
        {
            logger?.LogWarning("OpenAIConfigManager is not available for intent check.", nameof(LLMShaderGenerator));
            return false;
        }

        var client = configManager.Client;
        if (client == null)
        {
            logger?.LogWarning("OpenAI client is not available for intent check.", nameof(LLMShaderGenerator));
            return false;
        }

        string model = string.IsNullOrWhiteSpace(openAIIntentModelOverride)
            ? configManager.Model
            : openAIIntentModelOverride;

        try
        {
            var request = new ChatRequest(
                new[]
                {
                    new ChatMessage(Role.System, IntentSystemPrompt),
                    new ChatMessage(Role.User, transcription)
                },
                model: model,
                reasoningEffort: null
            );

            var response = await WithTimeout(
                client.ChatEndpoint.GetCompletionAsync(request), intentTimeoutSeconds, "OpenAI intent check");
            if (response?.FirstChoice?.Message?.Content != null)
            {
                string content = (string)response.FirstChoice.Message.Content;
                if (TryParseYesNo(content, out bool decision))
                {
                    return decision;
                }
            }
        }
        catch (Exception ex)
        {
            logger?.LogWarning($"OpenAI intent check failed: {ex.Message}", nameof(LLMShaderGenerator));
        }

        return false;
    }

    private async Task<bool> RunClaudeIntentCheckAsync(string transcription)
    {
        var configManager = claudeConfigManager ?? ClaudeConfigManager.Instance;
        if (configManager == null || !configManager.IsInitialized)
        {
            logger?.LogWarning("ClaudeConfigManager is not available for intent check.", nameof(LLMShaderGenerator));
            return false;
        }

        string model = string.IsNullOrWhiteSpace(claudeIntentModelOverride)
            ? configManager.ShaderModel
            : claudeIntentModelOverride;

        int maxTokens = intentMaxTokens > 0 ? intentMaxTokens : 16;

        var requestPayload = new ClaudeRequestBase
        {
            model = model,
            max_tokens = maxTokens,
            system = IntentSystemPrompt,
            messages = new[] { BuildClaudeUserMessage(transcription) }
        };

        try
        {
            string json = JsonUtility.ToJson(requestPayload);
            ClaudeRequestResult result = await SendClaudeRequestAsync(json, configManager, intentTimeoutSeconds);
            if (!result.success)
            {
                logger?.LogWarning($"Claude intent check failed: {result.transportError}", nameof(LLMShaderGenerator));
                return false;
            }

            string content = ExtractClaudeText(result.response);
            if (TryParseYesNo(content, out bool decision))
            {
                return decision;
            }
        }
        catch (Exception ex)
        {
            logger?.LogWarning($"Claude intent check failed: {ex.Message}", nameof(LLMShaderGenerator));
        }

        return false;
    }

    private static bool TryParseYesNo(string content, out bool decision)
    {
        decision = false;
        if (string.IsNullOrWhiteSpace(content))
        {
            return false;
        }

        string normalized = content.Trim().ToLowerInvariant();
        if (normalized.StartsWith("yes"))
        {
            decision = true;
            return true;
        }

        if (normalized.StartsWith("no"))
        {
            decision = false;
            return true;
        }

        if (normalized.StartsWith("true"))
        {
            decision = true;
            return true;
        }

        if (normalized.StartsWith("false"))
        {
            decision = false;
            return true;
        }

        return false;
    }

    private bool TryGetCachedIntentDecision(string transcription, out bool decision)
    {
        decision = false;
        if (intentCacheSize <= 0)
        {
            return false;
        }

        string key = NormalizeIntentCacheKey(transcription);
        if (string.IsNullOrWhiteSpace(key))
        {
            return false;
        }

        return intentCache.TryGetValue(key, out decision);
    }

    private void CacheIntentDecision(string transcription, bool decision)
    {
        if (intentCacheSize <= 0)
        {
            return;
        }

        string key = NormalizeIntentCacheKey(transcription);
        if (string.IsNullOrWhiteSpace(key))
        {
            return;
        }

        if (intentCache.ContainsKey(key))
        {
            intentCache[key] = decision;
            return;
        }

        intentCache[key] = decision;
        intentCacheOrder.Enqueue(key);

        while (intentCacheOrder.Count > intentCacheSize)
        {
            string oldestKey = intentCacheOrder.Dequeue();
            if (intentCache.ContainsKey(oldestKey))
            {
                intentCache.Remove(oldestKey);
            }
        }
    }

    private static string NormalizeIntentCacheKey(string transcription)
    {
        return string.IsNullOrWhiteSpace(transcription)
            ? string.Empty
            : transcription.Trim().ToLowerInvariant();
    }

    private static bool ContainsAnyKeyword(string input, IEnumerable<string> keywords)
    {
        foreach (var keyword in keywords)
        {
            if (input.IndexOf(keyword, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Parses the open-vocabulary detection manifest from header comments the LLM
    /// embeds in the generated shader source:
    ///   // @detect-classes: person, red coffee mug, laptop
    ///   // @detect-confidence: 0.3
    /// The class order defines the slot index used as the shader's labelID
    /// (slot 0 = first class). Returns false when no classes are declared (pure
    /// visual-effect shaders that do not need object detection).
    /// </summary>
    private static bool TryParseDetectionManifest(string shaderCode, out ShaderDetectionManifest manifest)
    {
        manifest = null;
        if (string.IsNullOrEmpty(shaderCode))
        {
            return false;
        }

        const string ClassesTag = "@detect-classes:";
        const string CocoTag = "@detect-coco:";
        const string ConfidenceTag = "@detect-confidence:";

        var classes = new List<string>();
        var cocoClasses = new List<string>();
        float confidence = 0.25f;

        foreach (var rawLine in shaderCode.Split('\n'))
        {
            string line = rawLine.Trim();

            int cocoIdx = line.IndexOf(CocoTag, StringComparison.OrdinalIgnoreCase);
            if (cocoIdx >= 0)
            {
                // Empty entries are KEPT, unlike the class list above: this list is positional
                // (index = slot), so dropping a blank would shift every later slot onto the
                // wrong class. "-" and "none" are the documented ways to say "no equivalent".
                string list = line.Substring(cocoIdx + CocoTag.Length);
                foreach (var part in list.Split(','))
                {
                    string name = part.Trim().Trim('"', '\'', ';').Trim();
                    if (name == "-" || string.Equals(name, "none", StringComparison.OrdinalIgnoreCase))
                    {
                        name = string.Empty;
                    }
                    cocoClasses.Add(name);
                }
                continue;
            }

            int classesIdx = line.IndexOf(ClassesTag, StringComparison.OrdinalIgnoreCase);
            if (classesIdx >= 0)
            {
                string list = line.Substring(classesIdx + ClassesTag.Length);
                foreach (var part in list.Split(','))
                {
                    string name = part.Trim().Trim('"', '\'', ';').Trim();
                    if (!string.IsNullOrEmpty(name))
                    {
                        classes.Add(name);
                    }
                }
                continue;
            }

            int confIdx = line.IndexOf(ConfidenceTag, StringComparison.OrdinalIgnoreCase);
            if (confIdx >= 0)
            {
                string value = line.Substring(confIdx + ConfidenceTag.Length).Trim().Trim(';').Trim();
                if (float.TryParse(value, System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out float parsed))
                {
                    confidence = Mathf.Clamp01(parsed);
                }
            }
        }

        if (classes.Count == 0)
        {
            return false;
        }

        // A COCO list that does not line up slot-for-slot is worse than none: it would point
        // slots at the wrong classes. Drop it and let the server fall back to its synonyms.
        if (cocoClasses.Count != classes.Count)
        {
            cocoClasses.Clear();
        }

        manifest = new ShaderDetectionManifest
        {
            classes = classes,
            cocoClasses = cocoClasses,
            confidence = confidence,
        };
        return true;
    }

    private struct ClaudeStreamResult
    {
        public bool success;
        public bool stalled;
        public long statusCode;
        public string error;
        public string text;
        public string stopReason;
        public string refusalCategory;
        public string servedModel;
    }

    [Serializable]
    private class ClaudeStreamEvent
    {
        public string type;
        public ClaudeStreamMessage message;
        public ClaudeStreamDelta delta;
        public ClaudeError error;
    }

    [Serializable]
    private class ClaudeStreamMessage
    {
        public string model;
    }

    [Serializable]
    private class ClaudeStreamDelta
    {
        public string type;
        public string text;
        public string stop_reason;
        public ClaudeStopDetails stop_details;
    }

    [Serializable]
    private class ClaudeStopDetails
    {
        public string category;
    }

    private struct ClaudeRequestResult
    {
        public bool success;
        public long statusCode;
        public string raw;
        public string transportError;
        public ClaudeResponse response;
    }

    [Serializable]
    private class ClaudeRequestBase
    {
        public string model;
        public int max_tokens;
        public string system;
        public ClaudeMessage[] messages;
    }


    [Serializable]
    private class ClaudeMessage
    {
        public string role;
        public ClaudeContent[] content;
    }

    [Serializable]
    private class ClaudeContent
    {
        public string type;
        public string text;
    }

    [Serializable]
    private class ClaudeResponse
    {
        public ClaudeContent[] content;
        public string stop_reason;
        public ClaudeError error;
    }

    [Serializable]
    private class ClaudeError
    {
        public string type;
        public string message;
    }
}
