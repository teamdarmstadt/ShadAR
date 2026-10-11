using System;
using UnityEngine;
using OpenAI;

/// <summary>
/// Manages OpenAI configuration with device-side override capability.
/// Loads configuration from two sources:
/// 1. Device persistent storage (higher priority) - JSON file that can be edited on device
/// 2. Resources folder (fallback) - baked into the APK
/// 
/// For each config field, if the device config has a non-empty value it's used,
/// otherwise the Resources config value is used as fallback.
/// </summary>
public class OpenAIConfigManager : MonoBehaviour
{
    private const string DefaultConfigFileName = "openai_config.json";
    private const string ResourcesConfigName = "OpenAIConfiguration";
    
    /// <summary>
    /// Configuration data structure for device-side JSON storage.
    /// </summary>
    [Serializable]
    private class DeviceConfig
    {
        public string apiKey = "";
        public string organizationId = "";
        public string proxyDomain = "";
        public string model = "";
        public string reasoningEffort = "";
    }

    /// <summary>
    /// Merged configuration values from device and Resources configs.
    /// </summary>
    private class MergedConfig
    {
        public string ApiKey { get; set; }
        public string OrganizationId { get; set; }
        public string ProxyDomain { get; set; }
        public string Model { get; set; }
        public string ReasoningEffort { get; set; }
        
        public bool HasValidApiKey => !string.IsNullOrWhiteSpace(ApiKey);
    }

    [Header("Model Configuration")]
    [Tooltip("Default model when the device config is empty. We use the most capable model available; older models work too, with more compile errors.")]
    [SerializeField] private string defaultModel = "gpt-6-astra";
    
    [Tooltip("Default reasoning effort level (None, Low, Medium, High). 'None' disables reasoning for non-reasoning models.")]
    [SerializeField] private string defaultReasoningEffort = "None";

    [Header("Device Configuration")]
    [Tooltip("Filename for device-side config in persistentDataPath")]
    [SerializeField] private string deviceConfigFileName = DefaultConfigFileName;

    [Header("Logger")]
    [SerializeField] private Logger logger;

    private string deviceConfigPath;
    private DeviceConfig deviceConfig;
    private OpenAIConfiguration resourcesConfig;
    private OpenAIClient openAIClient;
    private const string ModelPref = "PromptedVision.OpenAIModel";
    private const string EffortPref = "PromptedVision.OpenAIReasoningEffort";
    private bool isInitialized;

    /// <summary>
    /// Singleton instance of the OpenAI Config Manager.
    /// </summary>
    public static OpenAIConfigManager Instance { get; private set; }

    /// <summary>
    /// Gets whether the OpenAI client has been successfully initialized.
    /// </summary>
    public bool IsInitialized => isInitialized;
    
    /// <summary>
    /// Gets the initialized OpenAI client instance.
    /// </summary>
    public OpenAIClient Client => openAIClient;

    /// <summary>
    /// Gets the configured model name.
    /// </summary>
    public string Model { get; private set; }
    
    /// <summary>
    /// Gets the configured reasoning effort level ("None", "Low", "Medium", "High").
    /// </summary>
    public string ReasoningEffort { get; private set; }

    /// <summary>
    /// Initializes the singleton instance and loads configuration.
    /// </summary>
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

        logger?.LogDebug($"Device config path: {deviceConfigPath}", nameof(OpenAIConfigManager));
        
        InitializeConfiguration();
    }

    /// <summary>
    /// Called when the component is destroyed.
    /// </summary>
    private void OnDestroy()
    {
        if (Instance == this)
        {
            Instance = null;
        }
    }

    /// <summary>
    /// Initializes the OpenAI configuration by merging device and Resources configs.
    /// Device config values take priority; Resources config is used as fallback for empty fields.
    /// </summary>
    private void InitializeConfiguration()
    {
        deviceConfig = DeviceConfigStore.LoadOrCreate<DeviceConfig>(deviceConfigPath, logger, nameof(OpenAIConfigManager));
        LoadResourcesConfig();

        // Merge configs: device config values override Resources config values
        MergedConfig mergedConfig = CreateMergedConfig();
        
        LogMergedConfigSources(mergedConfig);

        if (!mergedConfig.HasValidApiKey)
        {
            logger?.LogError("No valid API key found in either device config or Resources config!", nameof(OpenAIConfigManager));
            return;
        }

        // Set the model and reasoning effort from merged config
        Model = mergedConfig.Model;
        string picked = PlayerPrefs.GetString(ModelPref, "");
        if (!string.IsNullOrWhiteSpace(picked)) Model = picked;
        ReasoningEffort = mergedConfig.ReasoningEffort;
        if (PlayerPrefs.HasKey(EffortPref)) ReasoningEffort = PlayerPrefs.GetString(EffortPref);
        logger?.LogDebug($"Using model: {Model}, reasoning effort: {ReasoningEffort}", nameof(OpenAIConfigManager));

        try
        {
            // Create authentication with merged values
            OpenAIAuthentication authentication = new OpenAIAuthentication(
                mergedConfig.ApiKey, 
                string.IsNullOrWhiteSpace(mergedConfig.OrganizationId) ? null : mergedConfig.OrganizationId
            );

            // Create settings with proxy domain if specified
            OpenAISettings settings = null;
            if (!string.IsNullOrWhiteSpace(mergedConfig.ProxyDomain))
            {
                settings = new OpenAISettings(mergedConfig.ProxyDomain);
                logger?.LogDebug($"Using proxy domain: {mergedConfig.ProxyDomain}", nameof(OpenAIConfigManager));
            }

            logger?.LogDebug("Creating OpenAI client...", nameof(OpenAIConfigManager));
            
            openAIClient = settings != null 
                ? new OpenAIClient(authentication, settings) 
                : new OpenAIClient(authentication);

            isInitialized = true;
            
            logger?.LogInfo("OpenAI client initialized successfully.", nameof(OpenAIConfigManager));
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to initialize OpenAI client: {ex.Message}", nameof(OpenAIConfigManager));
        }
    }

    /// <summary>
    /// Loads the OpenAI configuration from the Resources folder (baked into APK).
    /// </summary>
    private void LoadResourcesConfig()
    {
        try
        {
            resourcesConfig = Resources.Load<OpenAIConfiguration>(ResourcesConfigName);
            
            if (resourcesConfig != null)
            {
                logger?.LogDebug($"Resources config loaded: {ResourcesConfigName}", nameof(OpenAIConfigManager));
            }
            else
            {
                logger?.LogWarning($"No OpenAIConfiguration found in Resources folder with name: {ResourcesConfigName}", nameof(OpenAIConfigManager));
            }
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to load Resources config: {ex.Message}", nameof(OpenAIConfigManager));
            resourcesConfig = null;
        }
    }

    /// <summary>
    /// Creates a merged configuration from device and Resources configs.
    /// For each field: use device config value if not empty, otherwise use Resources config value.
    /// </summary>
    private MergedConfig CreateMergedConfig()
    {
        var merged = new MergedConfig();

        // Each field: device config value if set, else Resources/default fallback.
        merged.ApiKey = DeviceConfigStore.Merge(deviceConfig?.apiKey, resourcesConfig?.ApiKey);
        merged.OrganizationId = DeviceConfigStore.Merge(deviceConfig?.organizationId, resourcesConfig?.OrganizationId);
        merged.ProxyDomain = DeviceConfigStore.Merge(deviceConfig?.proxyDomain, resourcesConfig?.ProxyDomain);
        merged.Model = DeviceConfigStore.Merge(deviceConfig?.model, defaultModel);
        merged.ReasoningEffort = DeviceConfigStore.Merge(deviceConfig?.reasoningEffort, defaultReasoningEffort);

        return merged;
    }

    /// <summary>
    /// Logs which config source is being used for each field.
    /// </summary>
    private void LogMergedConfigSources(MergedConfig merged)
    {
        string apiKeySource = !string.IsNullOrWhiteSpace(deviceConfig?.apiKey) ? "Device" : 
                             !string.IsNullOrWhiteSpace(resourcesConfig?.ApiKey) ? "Resources" : "None";
        string orgIdSource = !string.IsNullOrWhiteSpace(deviceConfig?.organizationId) ? "Device" : 
                            !string.IsNullOrWhiteSpace(resourcesConfig?.OrganizationId) ? "Resources" : "None";
        string proxySource = !string.IsNullOrWhiteSpace(deviceConfig?.proxyDomain) ? "Device" : 
                            !string.IsNullOrWhiteSpace(resourcesConfig?.ProxyDomain) ? "Resources" : "None";
        string modelSource = !string.IsNullOrWhiteSpace(deviceConfig?.model) ? "Device" : "Default";
        string reasoningSource = !string.IsNullOrWhiteSpace(deviceConfig?.reasoningEffort) ? "Device" : "Default";

        logger?.LogInfo($"Config sources - ApiKey: {apiKeySource}, OrgId: {orgIdSource}, ProxyDomain: {proxySource}, Model: {modelSource}, ReasoningEffort: {reasoningSource}", nameof(OpenAIConfigManager));
    }

    /// <summary>
    /// Returns the path to the device config file for user reference.
    /// </summary>
    public string GetDeviceConfigPath()
    {
        return deviceConfigPath;
    }

    /// <summary>
    /// Model picked in the headset menu. Persisted, and wins over the config file so a model
    /// change needs no file edit.
    /// </summary>
    public void SetModel(string model)
    {
        Model = model;
        PlayerPrefs.SetString(ModelPref, model);
        PlayerPrefs.Save();
    }

    /// <summary>
    /// Reasoning effort picked in the headset menu ("None" = not sent). Persisted, and wins over
    /// the config file.
    /// </summary>
    public void SetReasoningEffort(string effort)
    {
        ReasoningEffort = effort;
        PlayerPrefs.SetString(EffortPref, effort);
        PlayerPrefs.Save();
    }
}
