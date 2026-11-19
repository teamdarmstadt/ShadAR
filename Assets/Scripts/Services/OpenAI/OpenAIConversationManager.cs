using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using OpenAI;
using OpenAI.Responses;

/// <summary>
/// Manages OpenAI conversation state for shader generation.
/// Maintains persistent conversation history to allow referencing previous shaders.
/// </summary>
public class OpenAIConversationManager : MonoBehaviour
{
    private const string ConversationStateFileName = "conversation_state.json";

    [Header("Configuration")]
    [Tooltip("Enable conversation mode to maintain history across shader generations")]
    [SerializeField] private bool enableConversationMode = true;

    [Tooltip("System prompt template file path (relative to StreamingAssets)")]
    [SerializeField] private string systemPromptPath = "Prompts/shader_prompt.txt";

    [Header("References")]
    [Tooltip("Reference to the OpenAI Config Manager (optional - will use singleton if not set)")]
    [SerializeField] private OpenAIConfigManager openAIConfigManager;

    [Header("Logger")]
    [SerializeField] private Logger logger;

    /// <summary>
    /// Serializable state for persisting conversation ID and whether prior turns used object detection.
    /// </summary>
    [Serializable]
    private class ConversationState
    {
        public string conversationId = "";
        public string lastUpdated = "";
        public int messageCount = 0;
        public bool usesObjectDetection = false;
    }

    private ConversationState state;
    private string stateFilePath;
    private OpenAI.Responses.Conversation currentConversation;
    private string persistentSystemPromptPath;
    private Task initializationTask;

    /// <summary>
    /// Singleton instance of the conversation manager.
    /// </summary>
    public static OpenAIConversationManager Instance { get; private set; }

    /// <summary>
    /// Whether conversation mode is configured on this manager (even if lazy initialization has not completed yet).
    /// </summary>
    public bool IsConversationEnabled => enableConversationMode;

    /// <summary>
    /// Gets whether conversation mode is enabled and initialized.
    /// </summary>
    public bool IsConversationModeActive => enableConversationMode && currentConversation != null;

    /// <summary>
    /// True when any turn in the current conversation used open-vocabulary object detection, so
    /// follow-up prompts ("make it more blue") retain the object-detection contract.
    /// </summary>
    public bool HasDetectionContext => state != null && state.usesObjectDetection;

    /// <summary>
    /// Event fired when conversation is created or updated.
    /// </summary>
    public event Action<OpenAI.Responses.Conversation> OnConversationUpdated;

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

        stateFilePath = Path.Combine(Application.persistentDataPath, ConversationStateFileName);
        persistentSystemPromptPath = Path.Combine(Application.persistentDataPath, systemPromptPath);

        LoadState();
    }

    private async void Start()
    {
        if (!enableConversationMode)
        {
            logger?.LogInfo("Conversation mode is disabled.", nameof(OpenAIConversationManager));
            return;
        }

        await EnsureConversationAsync();
    }

    private void OnDestroy()
    {
        if (Instance == this)
        {
            Instance = null;
        }
    }

    private void LoadState()
    {
        if (File.Exists(stateFilePath))
        {
            try
            {
                string json = File.ReadAllText(stateFilePath);
                state = JsonUtility.FromJson<ConversationState>(json) ?? new ConversationState();
                logger?.LogInfo($"Loaded conversation state. ID: {state.conversationId}, Messages: {state.messageCount}", nameof(OpenAIConversationManager));
            }
            catch (Exception ex)
            {
                logger?.LogError($"Failed to load conversation state: {ex.Message}", nameof(OpenAIConversationManager));
                state = new ConversationState();
            }
        }
        else
        {
            state = new ConversationState();
            logger?.LogInfo("No existing conversation state found. Will create new conversation.", nameof(OpenAIConversationManager));
        }
    }

    private void SaveState()
    {
        try
        {
            string json = JsonUtility.ToJson(state, true);
            File.WriteAllText(stateFilePath, json);
            logger?.LogDebug($"Saved conversation state to {stateFilePath}", nameof(OpenAIConversationManager));
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to save conversation state: {ex.Message}", nameof(OpenAIConversationManager));
        }
    }

    /// <summary>
    /// Ensures a valid OpenAI conversation exists, lazily retrying initialization if Start() ran
    /// before Android's StreamingAssets prompt copy completed.
    /// </summary>
    public async Task<OpenAI.Responses.Conversation> EnsureConversationAsync()
    {
        if (!enableConversationMode) return null;
        if (currentConversation != null) return currentConversation;

        if (initializationTask != null)
        {
            await initializationTask;
            return currentConversation;
        }

        initializationTask = InitializeConversation();
        try
        {
            await initializationTask;
        }
        finally
        {
            initializationTask = null;
        }

        return currentConversation;
    }

    private async Task InitializeConversation()
    {
        if (openAIConfigManager == null)
        {
            openAIConfigManager = OpenAIConfigManager.Instance;
        }

        if (openAIConfigManager == null || !openAIConfigManager.IsInitialized)
        {
            logger?.LogWarning("OpenAIConfigManager is not available or not initialized yet.", nameof(OpenAIConversationManager));
            return;
        }

        var client = openAIConfigManager.Client;
        if (client == null)
        {
            logger?.LogError("OpenAI client is not available!", nameof(OpenAIConversationManager));
            return;
        }

        try
        {
            if (!string.IsNullOrEmpty(state.conversationId))
            {
                logger?.LogDebug($"Attempting to retrieve conversation: {state.conversationId}", nameof(OpenAIConversationManager));

                try
                {
                    currentConversation = await client.ConversationsEndpoint.GetConversationAsync(state.conversationId);
                    logger?.LogInfo($"Retrieved existing conversation: {state.conversationId}", nameof(OpenAIConversationManager));
                    OnConversationUpdated?.Invoke(currentConversation);
                    return;
                }
                catch (Exception ex)
                {
                    logger?.LogWarning($"Failed to retrieve conversation {state.conversationId}: {ex.Message}. Creating new conversation.", nameof(OpenAIConversationManager));
                }
            }

            string systemPrompt = await GetSystemPromptAsync();
            if (string.IsNullOrEmpty(systemPrompt))
            {
                logger?.LogWarning("System prompt is not ready yet; will retry when generating.", nameof(OpenAIConversationManager));
                return;
            }

            logger?.LogDebug("Creating new conversation with system prompt...", nameof(OpenAIConversationManager));

            var createRequest = new CreateConversationRequest(new Message(Role.Developer, systemPrompt));
            currentConversation = await client.ConversationsEndpoint.CreateConversationAsync(createRequest);

            state.conversationId = currentConversation.Id;
            state.lastUpdated = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            state.messageCount = 0;
            state.usesObjectDetection = false;
            SaveState();

            logger?.LogInfo($"Created new conversation: {currentConversation.Id}", nameof(OpenAIConversationManager));
            OnConversationUpdated?.Invoke(currentConversation);
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to initialize conversation: {ex}", nameof(OpenAIConversationManager));
        }
    }

    /// <summary>
    /// Reads the base system prompt with multi-turn conversation instructions.
    /// </summary>
    public async Task<string> GetSystemPromptAsync()
    {
        if (!File.Exists(persistentSystemPromptPath))
        {
            string sourcePath = Path.Combine(Application.streamingAssetsPath, systemPromptPath);

#if UNITY_ANDROID && !UNITY_EDITOR
            logger?.LogWarning("System prompt not yet in persistent storage (waiting for StreamingAssets copy).", nameof(OpenAIConversationManager));
            return null;
#else
            if (File.Exists(sourcePath))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(persistentSystemPromptPath));
                File.Copy(sourcePath, persistentSystemPromptPath, false);
                logger?.LogInfo($"Copied system prompt to persistent storage: {persistentSystemPromptPath}", nameof(OpenAIConversationManager));
            }
            else
            {
                logger?.LogError($"System prompt file not found: {sourcePath}", nameof(OpenAIConversationManager));
                return null;
            }
#endif
        }

        try
        {
            string content = await File.ReadAllTextAsync(persistentSystemPromptPath);
            if (string.IsNullOrWhiteSpace(content))
            {
                logger?.LogError($"System prompt is empty: {persistentSystemPromptPath}", nameof(OpenAIConversationManager));
                return null;
            }

            content += "\n\nCONVERSATION MODE:\n";
            content += "- This is a multi-turn conversation about shader generation\n";
            content += "- User may reference \"previous shader\", \"last shader\", or \"the shader we just made\"\n";
            content += "- User may ask for modifications like \"make it more blue\", \"add a blur\", \"highlight cats instead\"\n";
            content += "- When modifying, maintain the structure and style of the previous shader\n";
            content += "- If starting fresh (no reference to previous), create a new shader from scratch\n";

            return content;
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to read system prompt: {ex.Message}", nameof(OpenAIConversationManager));
            return null;
        }
    }

    public OpenAI.Responses.Conversation GetConversation()
    {
        return currentConversation;
    }

    public string GetConversationId()
    {
        return currentConversation?.Id ?? state.conversationId;
    }

    public async Task ResetConversationAsync()
    {
        if (currentConversation != null && openAIConfigManager?.Client != null)
        {
            try
            {
                await openAIConfigManager.Client.ConversationsEndpoint.DeleteConversationAsync(currentConversation.Id);
                logger?.LogInfo($"Deleted conversation: {currentConversation.Id}", nameof(OpenAIConversationManager));
            }
            catch (Exception ex)
            {
                logger?.LogWarning($"Failed to delete conversation: {ex.Message}", nameof(OpenAIConversationManager));
            }
        }

        currentConversation = null;
        state = new ConversationState();
        SaveState();

        await EnsureConversationAsync();
    }

    public string GetConversationSummary()
    {
        if (currentConversation == null)
        {
            return "No active conversation.";
        }

        return $"Conversation ID: {currentConversation.Id}\n" +
               $"Messages: {state.messageCount}\n" +
               $"Last Updated: {state.lastUpdated}";
    }

    /// <summary>
    /// Updates conversation state after a completed turn.
    /// </summary>
    public void RecordTurn(bool usedObjectDetection)
    {
        state.messageCount++;
        state.usesObjectDetection |= usedObjectDetection;
        state.lastUpdated = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        SaveState();
    }

    public void IncrementMessageCount() => RecordTurn(false);
}
