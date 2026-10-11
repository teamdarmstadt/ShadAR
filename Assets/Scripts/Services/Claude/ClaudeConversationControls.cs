using UnityEngine;

/// <summary>
/// Utility component to manage Claude conversation from Unity Inspector or runtime.
/// Provides controls for resetting conversation and viewing status.
/// </summary>
public class ClaudeConversationControls : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private ClaudeConversationManager conversationManager;
    [SerializeField] private LLMShaderGenerator shaderGenerator;

    [Header("Logger")]
    [SerializeField] private Logger logger;

    [Header("Reset Input Configuration")]
    [Tooltip("Enable keyboard shortcut for resetting conversation")]
    [SerializeField] private bool enableKeyboardReset = false;

    [Tooltip("Primary key to trigger reset")]
    [SerializeField] private KeyCode resetKey = KeyCode.R;

    [Tooltip("Require modifier key (Ctrl/Cmd) to be held")]
    [SerializeField] private bool requireModifier = false;

    [Header("VR Input Configuration")]
    [Tooltip("Enable OVR button for resetting conversation")]
    [SerializeField] private bool enableVRReset = true;

    [Tooltip("OVR button to hold for reset (e.g., primary or secondary button)")]
    [SerializeField] private OVRInput.Button vrResetButton = OVRInput.Button.Two;

    [Tooltip("How long to hold the VR button to trigger reset (seconds)")]
    [SerializeField] private float vrHoldDuration = 2.0f;

    private float vrButtonHoldTime = 0f;

    [Header("Runtime Status (Read Only)")]
    [SerializeField, TextArea(3, 10)] private string conversationStatus;

    private void Start()
    {
        if (conversationManager == null)
        {
            conversationManager = ClaudeConversationManager.Instance;
        }

        TryResolveShaderGenerator();

        UpdateStatus();
    }

    private void Update()
    {
        bool providerMatch = IsProviderMatch();

        if (providerMatch && enableKeyboardReset)
        {
            bool modifierPressed = !requireModifier || Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);

            if (Input.GetKeyDown(resetKey) && modifierPressed)
            {
                string inputDesc = requireModifier ? $"Ctrl+{resetKey}" : resetKey.ToString();
                logger?.LogInfo($"{inputDesc} pressed - Resetting Claude conversation...", nameof(ClaudeConversationControls));
                ResetConversation();
            }
        }

        if (providerMatch && enableVRReset)
        {
            if (ConversationResetInput.AllHeld(vrResetButton))
            {
                vrButtonHoldTime += Time.deltaTime;

                if (vrButtonHoldTime >= vrHoldDuration)
                {
                    logger?.LogInfo($"VR button {vrResetButton} held for {vrHoldDuration}s - Resetting Claude conversation...", nameof(ClaudeConversationControls));
                    ResetConversation();
                    vrButtonHoldTime = 0f;
                }
            }
            else
            {
                vrButtonHoldTime = 0f;
            }
        }

        if (Time.frameCount % 60 == 0)
        {
            UpdateStatus();
        }
    }

    private void UpdateStatus()
    {
        if (!IsProviderMatch())
        {
            return;
        }

        if (conversationManager != null)
        {
            conversationStatus = conversationManager.GetConversationSummary();
        }
        else
        {
            conversationStatus = "ConversationManager not found.";
        }
    }

    [ContextMenu("Reset Conversation")]
    public async void ResetConversation()
    {
        if (!IsProviderMatch())
        {
            return;
        }

        if (conversationManager != null)
        {
            logger?.LogInfo("Resetting Claude conversation...", nameof(ClaudeConversationControls));
            await conversationManager.ResetConversationAsync();
            UpdateStatus();
            logger?.LogInfo("Claude conversation reset complete.", nameof(ClaudeConversationControls));
            logger?.SetStatus("Claude conversation cleared.");
            ConversationResetInput.Buzz();
        }
        else
        {
            logger?.LogWarning("Cannot reset - ClaudeConversationManager not found.", nameof(ClaudeConversationControls));
        }
    }

    [ContextMenu("Show Conversation Summary")]
    public void ShowSummary()
    {
        if (!IsProviderMatch())
        {
            return;
        }

        if (conversationManager != null)
        {
            string summary = conversationManager.GetConversationSummary();
            logger?.LogInfo($"Claude Conversation Summary:\n{summary}", nameof(ClaudeConversationControls));
            Debug.Log($"[ClaudeConversationControls] {summary}");
        }
    }

    private void TryResolveShaderGenerator()
    {
        if (shaderGenerator == null)
        {
            shaderGenerator = FindAnyObjectByType<LLMShaderGenerator>();
        }
    }

    private bool IsProviderMatch()
    {
        TryResolveShaderGenerator();

        if (shaderGenerator == null)
        {
            conversationStatus = "Controls disabled - LLMShaderGenerator not found.";
            return false;
        }

        if (shaderGenerator.SelectedProvider != LLMShaderGenerator.LlmProvider.Claude)
        {
            conversationStatus = $"Controls disabled - LLM provider is {shaderGenerator.SelectedProvider}.";
            return false;
        }

        return true;
    }
}
