using System;
using UnityEngine;

/// <summary>
/// Manages Claude configuration with device-side override capability.
/// Loads configuration from device persistent storage, falling back to Inspector defaults.
/// </summary>
public class ClaudeConfigManager : MonoBehaviour
{
    private const string DefaultConfigFileName = "claude_config.json";
    private const string ShaderModelPref = "PromptedVision.ClaudeShaderModel";
    private const string ShaderEffortPref = "PromptedVision.ClaudeShaderEffort";

    /// <summary>
    /// Overrides for shader generation. Empty/negative fields fall back to the global values.
    /// </summary>
    [Serializable]
    private class SectionConfig
    {
        public string model = "";
        public int maxTokens = -1;
        public string effort = "";     // low | medium | high | xhigh | max
        public string fallbacks = "";  // "default" = server-side refusal fallback, "none" = off
    }

    /// <summary>
    /// Configuration data structure for device-side JSON storage.
    /// The top-level model/maxTokens are the global defaults; the shader section overrides them.
    /// </summary>
    [Serializable]
    private class DeviceConfig
    {
        public string apiKey = "";
        public string model = "";
        public string baseUrl = "";
        public int maxTokens = -1;
        public SectionConfig shader = new();
    }

    [Header("Model Configuration (global defaults)")]
    [Tooltip("Default Claude model when the device config is empty. We use the most capable model " +
             "available; older models work too, with more compile errors.")]
    [SerializeField] private string defaultModel = "claude-fable-5-1";

    [Tooltip("Default max tokens per answer, thinking included. Requests are streamed, so a large " +
             "value does not risk an HTTP timeout.")]
    [SerializeField] private int defaultMaxTokens = 64000;

    [Tooltip("Default effort for shader generation (low, medium, high, xhigh, max; empty = model default). " +
             "Lower effort answers faster.")]
    [SerializeField] private string defaultShaderEffort = "high";

    [Tooltip("Let the API retry a declined request on a fallback model it picks per refusal category " +
             "(fallbacks: \"default\"). Turn off for models without server-side fallback.")]
    [SerializeField] private bool defaultServerSideFallbacks = true;

    [Tooltip("Claude Messages API endpoint")]
    [SerializeField] private string defaultBaseUrl = "https://api.anthropic.com/v1/messages";

    [Tooltip("Optional API key for editor/dev use. Prefer device config on target devices.")]
    [SerializeField] private string defaultApiKey = "";

    [Header("Device Configuration")]
    [Tooltip("Filename for device-side config in persistentDataPath")]
    [SerializeField] private string deviceConfigFileName = DefaultConfigFileName;

    [Header("Logger")]
    [SerializeField] private Logger logger;

    private string deviceConfigPath;
    private DeviceConfig deviceConfig;
    private bool isInitialized;

    /// <summary>
    /// Singleton instance of the Claude Config Manager.
    /// </summary>
    public static ClaudeConfigManager Instance { get; private set; }

    /// <summary>
    /// Gets whether the Claude configuration has been successfully initialized.
    /// </summary>
    public bool IsInitialized => isInitialized;

    /// <summary>
    /// Gets the configured API key.
    /// </summary>
    public string ApiKey { get; private set; }

    /// <summary>
    /// Gets the configured model name.
    /// </summary>
    public string Model { get; private set; }

    /// <summary>
    /// Gets the configured base URL.
    /// </summary>
    public string BaseUrl { get; private set; }

    /// <summary>
    /// Gets the configured max tokens.
    /// </summary>
    public int MaxTokens { get; private set; }

    /// <summary>
    /// Model for shader generation (device "shader" section, else the global model).
    /// </summary>
    public string ShaderModel { get; private set; }

    /// <summary>
    /// Max tokens for shader generation (device "shader" section, else the global max tokens).
    /// </summary>
    public int ShaderMaxTokens { get; private set; }

    /// <summary>
    /// Effort level for shader generation, or empty for the model's default.
    /// </summary>
    public string ShaderEffort { get; private set; }

    /// <summary>
    /// Whether shader requests opt into the server-side refusal fallback.
    /// </summary>
    public bool UseServerSideFallbacks { get; private set; }

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            enabled = false;
            Destroy(this);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);

        deviceConfigPath = DeviceConfigStore.ResolvePath(deviceConfigFileName);
        logger?.LogDebug($"Claude device config path: {deviceConfigPath}", nameof(ClaudeConfigManager));

        InitializeConfiguration();
    }

    private void OnDestroy()
    {
        if (Instance == this)
        {
            Instance = null;
        }
    }

    /// <summary>
    /// Initializes the Claude configuration by merging device and Inspector defaults.
    /// Device config values take priority; Inspector values are used as fallback.
    /// </summary>
    private void InitializeConfiguration()
    {
        deviceConfig = DeviceConfigStore.LoadOrCreate<DeviceConfig>(deviceConfigPath, logger, nameof(ClaudeConfigManager));

        ApiKey = DeviceConfigStore.Merge(deviceConfig.apiKey, defaultApiKey);
        Model = DeviceConfigStore.Merge(deviceConfig.model, defaultModel);
        BaseUrl = DeviceConfigStore.Merge(deviceConfig.baseUrl, defaultBaseUrl);
        MaxTokens = DeviceConfigStore.Merge(deviceConfig.maxTokens, defaultMaxTokens);

        // Per-purpose sections: section value if set, else the global default resolved above.
        ShaderModel = DeviceConfigStore.Merge(deviceConfig.shader?.model, Model);
        string picked = PlayerPrefs.GetString(ShaderModelPref, "");
        if (!string.IsNullOrWhiteSpace(picked)) ShaderModel = picked;
        ShaderMaxTokens = DeviceConfigStore.Merge(deviceConfig.shader?.maxTokens ?? -1, MaxTokens);
        ShaderEffort = DeviceConfigStore.Merge(deviceConfig.shader?.effort, defaultShaderEffort)?.Trim().ToLowerInvariant();
        // HasKey, not empty-check: "" is a real pick (model default) and must win over the file too.
        if (PlayerPrefs.HasKey(ShaderEffortPref)) ShaderEffort = PlayerPrefs.GetString(ShaderEffortPref);
        string fallbacks = deviceConfig.shader?.fallbacks?.Trim().ToLowerInvariant();
        UseServerSideFallbacks = string.IsNullOrEmpty(fallbacks) ? defaultServerSideFallbacks : fallbacks == "default";

        LogMergedConfigSources();

        if (string.IsNullOrWhiteSpace(ApiKey))
        {
            // Info, not an error: Claude is optional (the LLM switch falls back to OpenAI and greys
            // Claude out), so a missing key must not put an error into the status line.
            logger?.LogInfo($"No Claude API key - Claude is unavailable, OpenAI is used. Add one to {deviceConfigPath} to enable it.", nameof(ClaudeConfigManager));
            return;
        }

        if (string.IsNullOrWhiteSpace(BaseUrl))
        {
            logger?.LogError("Claude base URL is empty. Check device config or Inspector defaults.", nameof(ClaudeConfigManager));
            return;
        }

        isInitialized = true;
        logger?.LogInfo(
            $"Claude config initialized. Shader model: {ShaderModel} ({ShaderMaxTokens} tok, effort " +
            $"{(string.IsNullOrEmpty(ShaderEffort) ? "default" : ShaderEffort)}, fallbacks {(UseServerSideFallbacks ? "on" : "off")})",
            nameof(ClaudeConfigManager));
    }

    private void LogMergedConfigSources()
    {
        string apiKeySource = !string.IsNullOrWhiteSpace(deviceConfig?.apiKey) ? "Device" :
            !string.IsNullOrWhiteSpace(defaultApiKey) ? "Inspector" : "None";
        string modelSource = !string.IsNullOrWhiteSpace(deviceConfig?.model) ? "Device" : "Inspector";
        string baseUrlSource = !string.IsNullOrWhiteSpace(deviceConfig?.baseUrl) ? "Device" : "Inspector";
        string maxTokensSource = deviceConfig != null && deviceConfig.maxTokens > 0 ? "Device" : "Inspector";
        logger?.LogInfo(
            $"Claude config sources - ApiKey: {apiKeySource}, Model: {modelSource}, BaseUrl: {baseUrlSource}, MaxTokens: {maxTokensSource}",
            nameof(ClaudeConfigManager));
    }

    /// <summary>
    /// Returns the path to the device config file for user reference.
    /// </summary>
    public string GetDeviceConfigPath()
    {
        return deviceConfigPath;
    }

    /// <summary>
    /// Shader model picked in the headset menu. Persisted, and wins over the config file so a
    /// model change needs no file edit.
    /// </summary>
    public void SetShaderModel(string model)
    {
        ShaderModel = model;
        PlayerPrefs.SetString(ShaderModelPref, model);
        PlayerPrefs.Save();
    }

    /// <summary>
    /// Shader effort picked in the headset menu ("" = the model's default, nothing sent).
    /// Persisted, and wins over the config file.
    /// </summary>
    public void SetShaderEffort(string effort)
    {
        ShaderEffort = effort;
        PlayerPrefs.SetString(ShaderEffortPref, effort);
        PlayerPrefs.Save();
    }
}
