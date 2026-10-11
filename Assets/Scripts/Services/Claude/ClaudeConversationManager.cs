using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// Manages local conversation history for Claude shader generation.
/// Persists conversation messages to disk to allow multi-turn edits.
/// </summary>
public class ClaudeConversationManager : MonoBehaviour
{
    private const string ConversationStateFileName = "claude_conversation_state.json";

    [Header("Configuration")]
    [Tooltip("Enable conversation mode to maintain history across shader generations")]
    [SerializeField] private bool enableConversationMode = true;

    [Tooltip("System prompt template file path (relative to StreamingAssets)")]
    [SerializeField] private string systemPromptPath = "Prompts/shader_prompt.txt";

    [Tooltip("Maximum number of messages to keep in history (user + assistant). Set to 0 for unlimited.")]
    [SerializeField] private int maxMessageHistory = 20;

    [Header("Logger")]
    [SerializeField] private Logger logger;

    /// <summary>
    /// Serializable message entry for local conversation history.
    /// </summary>
    [Serializable]
    public class ConversationMessage
    {
        public string role = "";
        public string content = "";
    }

    /// <summary>
    /// Serializable state for persisting conversation history.
    /// </summary>
    [Serializable]
    private class ConversationState
    {
        public string lastUpdated = "";
        public int messageCount = 0;
        public List<ConversationMessage> messages = new();
    }

    private ConversationState state;
    private string stateFilePath;
    private string persistentSystemPromptPath;
    private string cachedSystemPrompt;

    /// <summary>
    /// Singleton instance of the conversation manager.
    /// </summary>
    public static ClaudeConversationManager Instance { get; private set; }

    /// <summary>
    /// Gets whether conversation mode is enabled.
    /// </summary>
    public bool IsConversationModeActive => enableConversationMode;

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
                state = JsonUtility.FromJson<ConversationState>(json) ?? new ConversationState();
                state.messages ??= new List<ConversationMessage>();
                logger?.LogInfo($"Loaded Claude conversation state. Messages: {state.messageCount}", nameof(ClaudeConversationManager));
            }
            catch (Exception ex)
            {
                logger?.LogError($"Failed to load Claude conversation state: {ex.Message}", nameof(ClaudeConversationManager));
                state = new ConversationState();
            }
        }
        else
        {
            state = new ConversationState();
            logger?.LogInfo("No existing Claude conversation state found. Starting fresh.", nameof(ClaudeConversationManager));
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
            logger?.LogDebug($"Saved Claude conversation state to {stateFilePath}", nameof(ClaudeConversationManager));
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to save Claude conversation state: {ex.Message}", nameof(ClaudeConversationManager));
        }
    }

    /// <summary>
    /// Reads the system prompt from persistent storage.
    /// Adds conversation instructions for multi-turn context.
    /// </summary>
    public async Task<string> GetSystemPromptAsync()
    {
        if (!string.IsNullOrWhiteSpace(cachedSystemPrompt))
        {
            return cachedSystemPrompt;
        }

        if (!File.Exists(persistentSystemPromptPath))
        {
            string sourcePath = Path.Combine(Application.streamingAssetsPath, systemPromptPath);

#if UNITY_ANDROID && !UNITY_EDITOR
            logger?.LogWarning("System prompt not found in persistent storage. Please ensure it's copied from StreamingAssets.", nameof(ClaudeConversationManager));
            return null;
#else
            if (File.Exists(sourcePath))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(persistentSystemPromptPath));
                File.Copy(sourcePath, persistentSystemPromptPath, false);
                logger?.LogInfo($"Copied system prompt to persistent storage: {persistentSystemPromptPath}", nameof(ClaudeConversationManager));
            }
            else
            {
                logger?.LogError($"System prompt file not found: {sourcePath}", nameof(ClaudeConversationManager));
                return null;
            }
#endif
        }

        try
        {
            string content = await File.ReadAllTextAsync(persistentSystemPromptPath);
            if (string.IsNullOrWhiteSpace(content))
            {
                logger?.LogError($"System prompt is empty: {persistentSystemPromptPath}", nameof(ClaudeConversationManager));
                return null;
            }

            content += "\n\nCONVERSATION MODE:\n";
            content += "- This is a multi-turn conversation about shader generation\n";
            content += "- User may reference \"previous shader\", \"last shader\", or \"the shader we just made\"\n";
            content += "- User may ask for modifications like \"make it more blue\", \"add a blur\", \"highlight cats instead\"\n";
            content += "- When modifying, maintain the structure and style of the previous shader\n";
            content += "- If starting fresh (no reference to previous), create a new shader from scratch\n";

            cachedSystemPrompt = content;
            return cachedSystemPrompt;
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to read system prompt: {ex.Message}", nameof(ClaudeConversationManager));
            return null;
        }
    }

    /// <summary>
    /// Gets a snapshot of the current conversation history.
    /// </summary>
    public List<ConversationMessage> GetMessagesSnapshot()
    {
        if (state == null || state.messages == null)
        {
            return new List<ConversationMessage>();
        }

        return new List<ConversationMessage>(state.messages);
    }

    /// <summary>
    /// Adds a user message to the conversation history.
    /// </summary>
    public void AddUserMessage(string message)
    {
        AddMessage("user", message);
    }

    /// <summary>
    /// Adds an assistant message to the conversation history.
    /// </summary>
    public void AddAssistantMessage(string message)
    {
        AddMessage("assistant", message);
    }

    private void AddMessage(string role, string content)
    {
        if (!enableConversationMode)
        {
            return;
        }

        if (state == null)
        {
            state = new ConversationState();
        }

        state.messages ??= new List<ConversationMessage>();

        if (string.IsNullOrWhiteSpace(content))
        {
            return;
        }

        state.messages.Add(new ConversationMessage { role = role, content = content });
        state.messageCount++;
        state.lastUpdated = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

        TrimHistory();
        SaveState();
    }

    private void TrimHistory()
    {
        if (maxMessageHistory <= 0 || state?.messages == null)
        {
            return;
        }

        if (state.messages.Count <= maxMessageHistory)
        {
            return;
        }

        int removeCount = state.messages.Count - maxMessageHistory;
        state.messages.RemoveRange(0, removeCount);

        // The Messages API requires the history to open with a user turn.
        while (state.messages.Count > 0 && state.messages[0].role != "user")
        {
            state.messages.RemoveAt(0);
        }
    }

    /// <summary>
    /// Clears the conversation history.
    /// </summary>
    public async Task ResetConversationAsync()
    {
        state = new ConversationState();
        cachedSystemPrompt = null;
        SaveState();
        await Task.CompletedTask;
        logger?.LogInfo("Claude conversation reset.", nameof(ClaudeConversationManager));
    }

    /// <summary>
    /// Gets conversation summary.
    /// </summary>
    public string GetConversationSummary()
    {
        if (!enableConversationMode)
        {
            return "Conversation mode disabled.";
        }

        int count = state?.messages?.Count ?? 0;
        return $"Claude Messages: {count}\nLast Updated: {state?.lastUpdated}";
    }
}
