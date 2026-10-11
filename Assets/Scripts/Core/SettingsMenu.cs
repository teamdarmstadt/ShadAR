using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using static UiKit;

/// <summary>
/// The settings panel (opened with Y), built at runtime into <see cref="menuPanel"/> in the
/// shared <see cref="UiKit"/> look.
///
/// **Display · Render** flips the passthrough between Stereo (each eye its own camera feed, true
/// depth) and Mono (both eyes show the left feed, zero disparity). That is the FEED, never the
/// OpenXR render mode — that stays Single Pass Instanced either way, because the Depth API
/// requires it and every generated shader carries the _EyeIndex gate regardless.
///
/// **LLM · Provider** picks who writes the shaders (OpenAI or Claude). Claude is optional: without
/// its API key the row is greyed out and OpenAI is used. **Model** lists what the provider's API
/// offers to this key, fetched live on open (once per provider and app run). **Effort** cycles the
/// effort level, "Default" first: that one sends no effort at all, the only setting every model
/// accepts.
///
/// **Shader · Detector** flips the detection server between YOLOE (open vocabulary: finds whatever
/// the active shader's @detect-classes list asks for) and yolo26-seg (closed COCO: faster, but only
/// the 80 COCO classes). **Clear conversation** starts the selected provider's conversation over,
/// the same as holding both triggers.
///
/// The menu holds no state of its own: <see cref="ShaderOverlayLoader"/> owns the render mode, the
/// detection provider the detector and <see cref="LLMShaderGenerator"/> the LLM settings, each
/// persisting its own value. Rows read them through getters, on open and after every press.
/// </summary>
public class SettingsMenu : MonoBehaviour, IMenu
{
    private const string OpenVocabDetector = "yoloe";
    private const string CocoDetector = "yolo";

    [Header("References (found in the scene when empty)")]
    [SerializeField] private ShaderOverlayLoader shaderOverlayLoader;
    [SerializeField] private LLMShaderGenerator shaderGenerator;
    [Tooltip("Detection provider to switch (DetectionNetworkClient). " +
             "Falls back to the ShaderOverlayLoader's provider when empty.")]
    [SerializeField] private MonoBehaviour detectionProviderBehaviour;
    [Tooltip("Both are asked; each only acts while its provider is the selected one.")]
    [SerializeField] private ClaudeConversationControls claudeConversation;
    [SerializeField] private OpenAIConversationControls openAIConversation;

    [Tooltip("The panel this menu builds into and shows/hides. It must be a DIFFERENT GameObject " +
             "than the one carrying this component: the panel starts hidden, and a component on a " +
             "hidden object gets no Update, so it could never poll the open button again.")]
    [SerializeField] private GameObject menuPanel;
    [Tooltip("A styled TMP_Dropdown in the scene, cloned for every choice row (its popup " +
             "template comes along). Kept hidden itself.")]
    [UnityEngine.Serialization.FormerlySerializedAs("modelDropdown")]
    [SerializeField] private TMP_Dropdown dropdownTemplate;

    [Header("Input")]
    [SerializeField] private OVRInput.Button openButton = OVRInput.Button.Four;

    [Header("Layout (design units; the panel's own scale sets the angular size)")]
    [SerializeField] private float panelWidth = 1000f;
    [SerializeField] private float rowHeight = 92f;
    [SerializeField] private float titleFontSize = 56f;
    [SerializeField] private float captionFontSize = 26f;
    [SerializeField] private float rowFontSize = 38f;

    private readonly List<(TMP_Text label, Func<string> value)> valueLabels = new();
    private readonly List<(Button button, CanvasGroup group, Func<bool> interactable)> conditionalRows = new();
    private readonly Dictionary<LLMShaderGenerator.LlmProvider, List<string>> fetchedModels = new();
    private RectTransform panelRect;
    private TMP_Dropdown modelDropdown;
    private List<string> shownModels = new();
    private ControllerTooltips.Hint hint;
    private bool built;

    private IDetectionProvider DetectionProvider =>
        (detectionProviderBehaviour as IDetectionProvider) ?? shaderOverlayLoader?.DetectionProvider;

    /// <summary>True while the panel is visible. Read by ControllerTooltips and the other menus.</summary>
    public bool IsMenuOpen => menuPanel != null && menuPanel.activeSelf;

    private void Awake()
    {
        // Self-hosting is a dead end: hiding the panel would stop this component's Update, so the
        // open button would never be polled again. Fail loudly instead of looking like a dead button.
        if (menuPanel == null || menuPanel == gameObject)
        {
            Debug.LogError($"{nameof(SettingsMenu)} on '{name}' needs menuPanel set to a SEPARATE panel " +
                           "GameObject. Put this component on the always-active canvas instead.", this);
            enabled = false;
            return;
        }

        if (shaderOverlayLoader == null) shaderOverlayLoader = FindAnyObjectByType<ShaderOverlayLoader>();
        if (shaderGenerator == null) shaderGenerator = FindAnyObjectByType<LLMShaderGenerator>();
        if (claudeConversation == null) claudeConversation = FindAnyObjectByType<ClaudeConversationControls>();
        if (openAIConversation == null) openAIConversation = FindAnyObjectByType<OpenAIConversationControls>();
        hint = ControllerTooltips.Register(openButton, "Settings", menu: this);
    }

    private void Start()
    {
        Build();
        menuPanel.SetActive(false);
    }

    private void OnDestroy() => ControllerTooltips.Unregister(hint);

    private void Update()
    {
        if (OVRInput.GetDown(openButton)) ToggleMenu();
    }

    // ------------------------------------------------------------------ open / close

    public void ToggleMenu()
    {
        if (IsMenuOpen)
        {
            CloseMenu();
            return;
        }

        Menus.CloseOthers(this);
        menuPanel.SetActive(true);
        RefreshAll(); // values can change elsewhere and survive a restart via PlayerPrefs
        RefreshModelDropdown();
    }

    public void CloseMenu()
    {
        if (menuPanel != null) menuPanel.SetActive(false);
    }

    private void RefreshAll()
    {
        foreach (var (label, value) in valueLabels) label.text = value();
        foreach (var (button, group, interactable) in conditionalRows) SetRowEnabled(button, group, interactable());
    }

    // ------------------------------------------------------------------ actions

    /// <summary>Flips the passthrough render mode (Stereo per-eye ↔ Mono) and persists it.</summary>
    public void ToggleStereoRendering()
    {
        if (shaderOverlayLoader == null) return;
        shaderOverlayLoader.SetStereoRendering(!shaderOverlayLoader.StereoRendering);
    }

    /// <summary>Flips the LLM provider that writes the shaders (OpenAI ↔ Claude) and persists it.</summary>
    public void ToggleLlmProvider()
    {
        if (shaderGenerator == null) return;

        shaderGenerator.SetProvider(shaderGenerator.SelectedProvider == LLMShaderGenerator.LlmProvider.OpenAI
            ? LLMShaderGenerator.LlmProvider.Claude
            : LLMShaderGenerator.LlmProvider.OpenAI);
        RefreshModelDropdown();
    }

    /// <summary>Steps the selected provider's effort to the next level, wrapping to Default.</summary>
    public void CycleEffort()
    {
        if (shaderGenerator == null) return;

        var choices = shaderGenerator.ShaderEffortChoices;
        int index = -1; // a config value the menu does not offer: the next press lands on Default
        for (int i = 0; i < choices.Count; i++)
        {
            if (choices[i] == shaderGenerator.CurrentShaderEffort) index = i;
        }
        shaderGenerator.SetShaderEffort(choices[(index + 1) % choices.Count]);
    }

    /// <summary>
    /// Flips the server-side detector between open-vocabulary YOLOE and closed COCO yolo26-seg.
    /// The first switch to YOLOE pays the MobileCLIP text-encoder load, so detections pause for a
    /// moment; flipping back is cheap, the server caches both models.
    /// </summary>
    public void ToggleDetector()
    {
        var provider = DetectionProvider;
        if (provider == null) return;

        // An empty ActiveDetector means "whatever the server was started with", which is the
        // open-vocabulary default, so the first press reliably lands on the COCO model.
        provider.SetDetector(provider.ActiveDetector != CocoDetector ? CocoDetector : OpenVocabDetector);
    }

    /// <summary>Clears the selected provider's conversation; confirmation is the status line + a buzz.</summary>
    public void ClearConversation()
    {
        claudeConversation?.ResetConversation();
        openAIConversation?.ResetConversation();
    }

    // ------------------------------------------------------------------ building

    private void Build()
    {
        if (built) return;
        built = true;

        panelRect = (RectTransform)menuPanel.transform;
        panelRect.sizeDelta = new Vector2(panelWidth, panelRect.sizeDelta.y);

        var plate = Ensure<Image>(menuPanel);
        plate.color = PanelColor;
        plate.raycastTarget = true; // the plate catches stray hits so they do not fall through

        var layout = Ensure<VerticalLayoutGroup>(menuPanel);
        layout.padding = new RectOffset(48, 48, 36, 44);
        layout.spacing = 12f;
        layout.childControlWidth = layout.childControlHeight = true;
        layout.childForceExpandWidth = true;
        layout.childForceExpandHeight = false;
        Ensure<ContentSizeFitter>(menuPanel).verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        if (dropdownTemplate != null)
        {
            // The template itself is not a row; it must not take part in the layout.
            dropdownTemplate.gameObject.SetActive(false);
            Ensure<LayoutElement>(dropdownTemplate.gameObject).ignoreLayout = true;
        }

        BuildHeader();

        Caption("DISPLAY");
        ValueRow("Render", () => shaderOverlayLoader != null && !shaderOverlayLoader.StereoRendering ? "Mono" : "Stereo",
            ToggleStereoRendering);

        Caption("LLM");
        // Greyed out without a Claude key: OpenAI is then the only choice.
        ValueRow("Provider", () => shaderGenerator == null ? "-"
                : !shaderGenerator.ClaudeAvailable ? "OpenAI (no Claude key)"
                : shaderGenerator.SelectedProvider.ToString(),
            ToggleLlmProvider, interactable: () => shaderGenerator != null && shaderGenerator.ClaudeAvailable);
        modelDropdown = ChoiceRow("Model", i =>
        {
            if (i >= 0 && i < shownModels.Count) shaderGenerator?.SetShaderModel(shownModels[i]);
        });
        ValueRow("Effort", () =>
        {
            string effort = shaderGenerator != null ? shaderGenerator.CurrentShaderEffort : null;
            return string.IsNullOrEmpty(effort) ? "Default" : effort;
        }, CycleEffort);

        Caption("SHADER");
        ValueRow("Detector", () => DetectionProvider?.ActiveDetector == CocoDetector ? "COCO fast" : "Open Vocab",
            ToggleDetector);
        ActionRow("Clear conversation", ClearConversation);
    }

    private void BuildHeader()
    {
        var header = NewRect("Header", panelRect);
        Size(header, 80f);

        var title = Label(header, "Title", titleFontSize, TextAlignmentOptions.Left);
        Stretch((RectTransform)title.transform);
        title.text = "Settings";
        title.fontStyle = FontStyles.Bold;

        // Close button for ray/poke users; Y does the same.
        var close = Button(header, "Close", RowColor, CloseMenu);
        var closeRect = (RectTransform)close.transform;
        closeRect.anchorMin = closeRect.anchorMax = closeRect.pivot = new Vector2(1f, 0.5f);
        closeRect.sizeDelta = new Vector2(80f, 80f);
        var x = Label(closeRect, "X", rowFontSize, TextAlignmentOptions.Center);
        Stretch((RectTransform)x.transform);
        x.text = "×";
    }

    private void Caption(string text)
    {
        var caption = Label(panelRect, "Caption", captionFontSize, TextAlignmentOptions.BottomLeft);
        caption.text = text;
        caption.color = CaptionColor;
        caption.characterSpacing = 8f;
        Size((RectTransform)caption.transform, 52f);
    }

    /// <summary>A row showing <paramref name="value"/> on the right; greyed out while <paramref name="interactable"/> is false.</summary>
    private void ValueRow(string name, Func<string> value, Action onClick, Func<bool> interactable = null)
    {
        var button = Button(panelRect, name, RowColor, () =>
        {
            onClick();
            RefreshAll();
        });
        var rect = (RectTransform)button.transform;
        Size(rect, rowHeight);

        var label = Label(rect, "Name", rowFontSize, TextAlignmentOptions.Left);
        Stretch((RectTransform)label.transform, 32f);
        label.text = name;

        var valueLabel = Label(rect, "Value", rowFontSize, TextAlignmentOptions.Right);
        Stretch((RectTransform)valueLabel.transform, 32f);
        valueLabel.color = ValueColor;
        valueLabel.text = value();
        valueLabels.Add((valueLabel, value));

        if (interactable != null)
        {
            // The disabled button only dims its background; the group dims the labels too.
            var group = button.gameObject.AddComponent<CanvasGroup>();
            SetRowEnabled(button, group, interactable());
            conditionalRows.Add((button, group, interactable));
        }
    }

    private static void SetRowEnabled(Button button, CanvasGroup group, bool enabled)
    {
        button.interactable = enabled;
        group.alpha = enabled ? 1f : 0.45f;
    }

    private void ActionRow(string name, Action onClick)
    {
        var button = Button(panelRect, name, RowColor, () => onClick());
        Size((RectTransform)button.transform, rowHeight);
        var label = Label((RectTransform)button.transform, "Name", rowFontSize, TextAlignmentOptions.Center);
        Stretch((RectTransform)label.transform);
        label.text = name;
    }

    private TMP_Dropdown ChoiceRow(string name, Action<int> onPick)
    {
        var row = NewRect(name, panelRect);
        Size(row, rowHeight);
        row.gameObject.AddComponent<Image>().color = RowColor;

        var label = Label(row, "Name", rowFontSize, TextAlignmentOptions.Left);
        Stretch((RectTransform)label.transform, 32f);
        label.text = name;

        if (dropdownTemplate == null)
        {
            Debug.LogError($"{nameof(SettingsMenu)}: no dropdownTemplate assigned; choice row '{name}' stays empty.", this);
            return null;
        }

        var dropdown = Instantiate(dropdownTemplate, row, false);
        dropdown.name = "Dropdown";
        dropdown.gameObject.SetActive(true);
        dropdown.GetComponent<LayoutElement>().ignoreLayout = false;
        dropdown.onValueChanged.RemoveAllListeners();
        dropdown.onValueChanged.AddListener(i => onPick(i));
        Style(dropdown);

        var rect = (RectTransform)dropdown.transform;
        rect.anchorMin = new Vector2(0.32f, 0f);
        rect.anchorMax = Vector2.one;
        rect.offsetMin = new Vector2(0f, 10f);
        rect.offsetMax = new Vector2(-10f, -10f);
        return dropdown;
    }

    private void Style(TMP_Dropdown dropdown)
    {
        dropdown.image.color = Color.white;
        dropdown.colors = RowColors(new Color(0.10f, 0.11f, 0.14f, 1f));
        if (dropdown.captionText != null)
        {
            dropdown.captionText.color = ValueColor;
            dropdown.captionText.fontSize = rowFontSize - 4f;
            dropdown.captionText.alignment = TextAlignmentOptions.Right;
            dropdown.captionText.textWrappingMode = TextWrappingModes.NoWrap;
            dropdown.captionText.overflowMode = TextOverflowModes.Ellipsis;
        }
        var arrow = dropdown.transform.Find("Arrow")?.GetComponent<Image>();
        if (arrow != null) arrow.color = ValueColor;

        // The popup list is cloned from the template on open, so styling the template is enough.
        if (dropdown.template != null)
        {
            var templateImage = dropdown.template.GetComponent<Image>();
            if (templateImage != null) templateImage.color = new Color(0.08f, 0.09f, 0.11f, 0.98f);
            foreach (var toggle in dropdown.template.GetComponentsInChildren<Toggle>(true))
            {
                toggle.colors = RowColors(RowColor);
                if (toggle.targetGraphic != null) toggle.targetGraphic.color = Color.white;
                if (toggle.graphic != null) toggle.graphic.color = AccentColor;
            }
        }
        if (dropdown.itemText != null)
        {
            dropdown.itemText.color = Color.white;
            dropdown.itemText.fontSize = rowFontSize - 6f;
        }
    }

    // ------------------------------------------------------------------ model list

    /// <summary>
    /// Shows the current model right away, then the provider's live list once it has arrived.
    /// A failed fetch keeps just the current model and is retried on the next open.
    /// </summary>
    private async void RefreshModelDropdown()
    {
        if (modelDropdown == null || shaderGenerator == null) return;

        var provider = shaderGenerator.SelectedProvider;
        ShowModels(null);

        if (!fetchedModels.ContainsKey(provider))
        {
            // Greyed out while loading, so a one-entry list reads as "not there yet", not "that's all".
            modelDropdown.interactable = false;
            var models = await shaderGenerator.ListShaderModelsAsync();
            modelDropdown.interactable = true;
            if (models == null || models.Count == 0) return;
            fetchedModels[provider] = models;
        }

        // The provider may have been flipped while the list was loading.
        if (provider == shaderGenerator.SelectedProvider) ShowModels(fetchedModels[provider]);
    }

    private void ShowModels(List<string> models)
    {
        string current = shaderGenerator.CurrentShaderModel ?? "(not configured)";
        shownModels = models != null ? new List<string>(models) : new List<string>();

        // A model from the config file that the list does not offer stays selectable as-is.
        if (!shownModels.Contains(current)) shownModels.Insert(0, current);

        modelDropdown.ClearOptions();
        modelDropdown.AddOptions(shownModels);
        modelDropdown.SetValueWithoutNotify(shownModels.IndexOf(current));
    }
}
