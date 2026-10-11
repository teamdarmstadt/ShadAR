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
    /// Serializable state for persisting conversation ID.
    /// </summary>
    [Serializable]
    private class ConversationState
    {
        public string conversationId = "";
        public string lastUpdated = "";
        public int messageCount = 0;
    }
    
    private ConversationState state;
    private string stateFilePath;
    private OpenAI.Responses.Conversation currentConversation;
    private string persistentSystemPromptPath;
    
    /// <summary>
    /// Singleton instance of the conversation manager.
    /// </summary>
    public static OpenAIConversationManager Instance { get; private set; }
    
    /// <summary>
    /// Gets whether conversation mode is enabled and initialized.
    /// </summary>
    public bool IsConversationModeActive => enableConversationMode && currentConversation != null;
    
    /// <summary>
    /// Event fired when conversation is created or updated.
    /// </summary>
    public event Action<OpenAI.Responses.Conversation> OnConversationUpdated;
    
    /// <summary>
    /// Initializes the singleton and loads conversation state.
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
        
        stateFilePath = Path.Combine(Application.persistentDataPath, ConversationStateFileName);
        persistentSystemPromptPath = Path.Combine(Application.persistentDataPath, systemPromptPath);

        LoadState();
    }
    
    /// <summary>
    /// Initializes the conversation when the component starts.
    /// </summary>
    private async void Start()
    {
        if (!enableConversationMode)
        {
            logger?.LogInfo("Conversation mode is disabled.", nameof(OpenAIConversationManager));
            return;
        }
        
        // Use singleton instance if no direct reference is set
        if (openAIConfigManager == null)
        {
            openAIConfigManager = OpenAIConfigManager.Instance;
        }
        
        if (openAIConfigManager == null || !openAIConfigManager.IsInitialized)
        {
            logger?.LogError("OpenAIConfigManager is not available or not initialized!", nameof(OpenAIConversationManager));
            enableConversationMode = false;
            return;
        }
        
        await InitializeConversation();
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
    /// Loads conversation state from disk.
    /// </summary>
    private void LoadState()
    {
        if (File.Exists(stateFilePath))
        {
            try
            {
                string json = File.ReadAllText(stateFilePath);
                state = JsonUtility.FromJson<ConversationState>(json);
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
    
    /// <summary>
    /// Saves conversation state to disk.
    /// </summary>
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
    /// Initializes or retrieves the conversation.
    /// </summary>
    private async Task InitializeConversation()
    {
        var client = openAIConfigManager.Client;
        
        if (client == null)
        {
            logger?.LogError("OpenAI client is not available!", nameof(OpenAIConversationManager));
            return;
        }
        
        try
        {
            // Try to retrieve existing conversation
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
            
            // Create new conversation with system prompt
            string systemPrompt = await ReadSystemPrompt();
            if (string.IsNullOrEmpty(systemPrompt))
            {
                logger?.LogError("System prompt is empty. Cannot create conversation.", nameof(OpenAIConversationManager));
                return;
            }
            
            logger?.LogDebug("Creating new conversation with system prompt...", nameof(OpenAIConversationManager));
            
            var createRequest = new CreateConversationRequest(new Message(Role.Developer, systemPrompt));
            currentConversation = await client.ConversationsEndpoint.CreateConversationAsync(createRequest);
            
            state.conversationId = currentConversation.Id;
            state.lastUpdated = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            state.messageCount = 0;
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
    /// Reads the system prompt from persistent storage.
    /// </summary>
    private async Task<string> ReadSystemPrompt()
    {
        // Copy from StreamingAssets if needed
        if (!File.Exists(persistentSystemPromptPath))
        {
            string sourcePath = Path.Combine(Application.streamingAssetsPath, systemPromptPath);
            
#if UNITY_ANDROID && !UNITY_EDITOR
            // On Android, we need to use UnityWebRequest
            logger?.LogWarning("System prompt not found in persistent storage. Please ensure it's copied from StreamingAssets.", nameof(OpenAIConversationManager));
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
            
            // Enhance system prompt with conversation context
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
    
    /// <summary>
    /// Gets the current conversation object for use in API calls.
    /// </summary>
    public OpenAI.Responses.Conversation GetConversation()
    {
        return currentConversation;
    }
    
    /// <summary>
    /// Gets the conversation ID.
    /// </summary>
    public string GetConversationId()
    {
        return currentConversation?.Id ?? state.conversationId;
    }
    
    /// <summary>
    /// Clears the current conversation and creates a new one.
    /// </summary>
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
        
        await InitializeConversation();
    }
    
    /// <summary>
    /// Gets conversation summary.
    /// </summary>
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
    /// Updates the message count after receiving a response.
    /// </summary>
    public void IncrementMessageCount()
    {
        state.messageCount++;
        state.lastUpdated = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        SaveState();
    }
}
