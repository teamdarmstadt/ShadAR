using UnityEngine;
using UnityEngine.UI;
using System.Text.RegularExpressions;
using Networking;

/// <summary>
/// Manages the UI menu for browsing and selecting shader history.
/// Provides VR-friendly navigation using OVR input controls.
/// </summary>
public class ShaderHistoryMenu : MonoBehaviour, IMenu
{
    private const string TimestampFormat = "yyyyMMdd_HHmmss";
    private const string DisplayDateFormat = "yyyy-MM-dd HH:mm:ss";
    private const string ParsePattern = @"^(.*?)(?:_([0-9]{8}_[0-9]{6}))?$";
    
    // Same palette as the settings panel (UiKit), so both read as one interface.
    private static Color HighlightColor => UiKit.AccentColor;
    private static Color DefaultColor => UiKit.RowColor;

    [Header("References")]
    [SerializeField] private ShaderOverlayLoader shaderOverlayLoader;
    [SerializeField] private GameObject shaderHistoryMenu;
    [SerializeField] private RectTransform contentParent;
    [Tooltip("Prefab for one grid tile (Button root + RawImage preview + name/date labels)")]
    [SerializeField] private GameObject tilePrefab;

    [Header("Input Settings")]
    [Tooltip("Opens/closes the shader history menu. Tiles are picked by controller ray + trigger " +
             "or finger poke — no thumbstick navigation.")]
    [SerializeField] private OVRInput.Button openButton;

    [Header("WebSocket Client")]
    [SerializeField] private ShaderBundleWebSocketClient webSocketClient;

    [Header("Scroll Settings")]
    [SerializeField] private ScrollRect scrollRect;
    [Tooltip("Ensure the scroll content has a vertical ContentSizeFitter so it grows with " +
             "the tiles and can actually scroll. Disable if you size the content yourself.")]
    [SerializeField] private bool autoSizeScrollContent = true;

    [Header("Live Preview")]
    [Tooltip("Show the live camera feed with each shader applied in every tile")]
    [SerializeField] private bool enableLivePreviews = true;
    [Tooltip("Preview render width (height follows the camera aspect)")]
    [Range(64, 512)] [SerializeField] private int previewResolution = 256;
    [Tooltip("How many tiles to refresh per frame (lower = cheaper, still looks live)")]
    [Range(1, 8)] [SerializeField] private int previewsPerFrame = 2;

    [Header("Selection Visuals")]
    [Tooltip("Preview tint of the selected tile (full brightness)")]
    [SerializeField] private Color selectedTint = Color.white;
    [Tooltip("Preview tint of non-selected tiles (dimmed)")]
    [SerializeField] private Color unselectedTint = new(0.5f, 0.5f, 0.5f, 1f);
    [Tooltip("Scale bump applied to the selected tile (1 = none)")]
    [SerializeField] private float selectedScale = 1.05f;

    [Header("Selected-Tile Detections")]
    [Tooltip("Render the SELECTED tile with real object detections. While a tile is " +
             "selected its vocabulary is activated on the detector; other tiles stay " +
             "passthrough. Note: this changes the live detector vocabulary while browsing.")]
    [SerializeField] private bool previewSelectedWithDetections = true;
    [Tooltip("Detection provider (e.g. DetectionNetworkClient). Falls back to the " +
             "ShaderOverlayLoader's provider when empty.")]
    [SerializeField] private MonoBehaviour detectionProviderBehaviour;
    [Tooltip("Seconds the cursor must rest on a tile before its vocabulary is pushed")]
    [SerializeField] private float vocabularySettleDelay = 0.25f;
    [Tooltip("Extra wait after pushing the vocabulary before showing detections, so the " +
             "selected tile never displays stale detections from the previous shader (RTT)")]
    [SerializeField] private float detectionActivationGrace = 0.4f;

    private int selectedIndex;
    private ShaderPreviewRenderer previewRenderer;

    // Debounced per-shader vocabulary activation for the selected-tile detection preview.
    private float selectionChangedTime;
    private bool vocabularyPushPending;
    private bool vocabularyApplied;
    private float vocabularyAppliedTime;
    private bool selectedDetectionsActive;
    private System.Action<Material> bindSelectedDetectionsCached;

    private IDetectionProvider PreviewDetectionProvider =>
        (detectionProviderBehaviour as IDetectionProvider) ?? shaderOverlayLoader?.DetectionProvider;


    /// <summary>
    /// Initializes the menu on start.
    /// </summary>
    private ControllerTooltips.Hint tooltip;

    private void Start()
    {
        tooltip = ControllerTooltips.Register(openButton, "Shader History", menu: this);
        bindSelectedDetectionsCached = BindSelectedDetections;

        PopulateMenu();
    }

    /// <summary>
    /// Handles menu toggle and navigation input.
    /// </summary>
    private void Update()
    {
        if (OVRInput.GetDown(openButton))
        {
            ToggleMenu();
        }
        
        if (shaderHistoryMenu.activeSelf)
        {
            if (enableLivePreviews && previewRenderer != null && shaderOverlayLoader != null)
            {
                if (previewSelectedWithDetections)
                {
                    float now = Time.unscaledTime;

                    // 1) Push the selected tile's vocabulary once the cursor has settled.
                    if (vocabularyPushPending && now - selectionChangedTime >= vocabularySettleDelay)
                    {
                        PushSelectedVocabulary();
                        vocabularyPushPending = false;
                        vocabularyApplied = true;
                        vocabularyAppliedTime = now;
                    }

                    // 2) Only show detections after the new vocabulary has had time to take
                    //    effect (one detection round-trip), so no stale effects leak in.
                    if (vocabularyApplied && !selectedDetectionsActive
                        && now - vocabularyAppliedTime >= detectionActivationGrace)
                    {
                        selectedDetectionsActive = true;
                    }
                }

                previewRenderer.Tick(shaderOverlayLoader.GetCurrentCameraFeed(), bindSelectedDetectionsCached);
            }
        }
    }

    /// <summary>
    /// Subscribes to bundle received event.
    /// </summary>
    private void OnEnable()
    {
        if (webSocketClient != null)
        {
            webSocketClient.OnBundleReceived += OnBundleReceived;
        }
    }

    /// <summary>
    /// Unsubscribes from bundle received event.
    /// </summary>
    private void OnDisable()
    {
        if (webSocketClient != null)
        {
            webSocketClient.OnBundleReceived -= OnBundleReceived;
        }
    }

    /// <summary>
    /// Releases preview render textures and material copies.
    /// </summary>
    private void OnDestroy()
    {
        ControllerTooltips.Unregister(tooltip);
        previewRenderer?.Dispose();
        previewRenderer = null;
    }

    /// <summary>
    /// Number of selectable tiles, capped to the tiles that actually exist in the content
    /// so navigation can never land on a phantom entry past the real ones.
    /// </summary>
    private int GetSelectableCount()
    {
        int byHistory = shaderOverlayLoader.GetShaderCount() + 1; // shaders + the "None" tile
        int byChildren = contentParent != null ? contentParent.childCount : byHistory;
        return Mathf.Max(0, Mathf.Min(byHistory, byChildren));
    }

    /// <summary>
    /// Number of tiles per row for navigation — read from the GridLayoutGroup the user
    /// configured. Falls back to 1 (vertical stepping) when no fixed-column grid exists.
    /// </summary>
    private int GetColumnCount()
    {
        if (contentParent != null
            && contentParent.TryGetComponent<GridLayoutGroup>(out var grid)
            && grid.constraint == GridLayoutGroup.Constraint.FixedColumnCount
            && grid.constraintCount > 0)
        {
            return grid.constraintCount;
        }
        return 1;
    }

    /// <summary>
    /// Populates the menu with shader history entries.
    /// </summary>
    private void PopulateMenu()
    {
        shaderOverlayLoader.RefreshShaderHistory(() => PopulateMenuInternal());
    }

    /// <summary>
    /// Internal method to create menu rows.
    /// </summary>
    private void PopulateMenuInternal()
    {
        ClearMenuContent();
        EnsureScrollContentFitter();
        EnsurePreviewRenderer();
        // History may have changed (new/removed bundles) — drop stale preview slots so
        // they are rebuilt against the freshly loaded materials.
        previewRenderer?.ClearSlots();
        CreateNoneTile();
        CreateShaderTiles();
        UpdateTileHighlight();
    }

    /// <summary>
    /// Ensures the scroll content grows vertically with its tiles (via a ContentSizeFitter)
    /// so the ScrollRect actually has room to scroll to the last entries. Only touches the
    /// fitter — the grid layout and tile prefab are left to the user's configuration.
    /// </summary>
    private void EnsureScrollContentFitter()
    {
        if (!autoSizeScrollContent || scrollRect == null) return;

        var content = scrollRect.content != null ? scrollRect.content : contentParent;
        if (content == null) return;

        if (!content.TryGetComponent<ContentSizeFitter>(out var fitter))
        {
            fitter = content.gameObject.AddComponent<ContentSizeFitter>();
        }
        fitter.horizontalFit = ContentSizeFitter.FitMode.Unconstrained;
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
    }

    /// <summary>
    /// Lazily creates the live-preview renderer and applies the throttle setting.
    /// </summary>
    private void EnsurePreviewRenderer()
    {
        if (!enableLivePreviews) return;
        previewRenderer ??= new ShaderPreviewRenderer(previewResolution, GetCameraAspect());
        previewRenderer.PreviewsPerFrame = previewsPerFrame;
    }

    /// <summary>
    /// Aspect (width / height) of the live camera feed, so preview textures are not
    /// distorted. Falls back to 4:3 when the feed isn't available yet.
    /// </summary>
    private float GetCameraAspect()
    {
        var feed = shaderOverlayLoader != null ? shaderOverlayLoader.GetCurrentCameraFeed() : null;
        if (feed != null && feed.height > 0)
        {
            return (float)feed.width / feed.height;
        }
        return 4f / 3f;
    }

    /// <summary>
    /// Assigns the live-preview texture to the cell's RawImage child. Layout of the
    /// RawImage (and everything else in the tile) is left entirely to the prefab.
    /// </summary>
    private static void AssignPreview(GameObject cell, Texture texture)
    {
        if (texture == null) return;
        var raw = cell.GetComponentInChildren<RawImage>(true);
        if (raw != null)
        {
            raw.texture = texture;
        }
    }

    /// <summary>
    /// Clears all existing menu rows.
    /// </summary>
    private void ClearMenuContent()
    {
        // Detach each child immediately before destroying it. Destroy() is deferred to the
        // end of the frame, so without detaching, the old tiles would still count toward
        // childCount/layout while the new ones are created — inflating the content height
        // and letting the cursor land on phantom tiles.
        for (int i = contentParent.childCount - 1; i >= 0; i--)
        {
            var child = contentParent.GetChild(i);
            child.SetParent(null, false);
            Destroy(child.gameObject);
        }
    }

    /// <summary>
    /// Creates the "None" tile for resetting shader.
    /// </summary>
    private void CreateNoneTile()
    {
        var noneTile = Instantiate(tilePrefab, contentParent);
        var noneColumns = noneTile.GetComponentsInChildren<TMPro.TextMeshProUGUI>();

        if (noneColumns.Length >= 1) noneColumns[0].text = "None";
        if (noneColumns.Length >= 2) noneColumns[1].text = "Passthrough";

        if (noneTile.TryGetComponent<Button>(out var noneButton))
        {
            // Closes like every other tile: left open, the highlight stays on the previous
            // shader and the click looks like it did nothing.
            noneButton.onClick.AddListener(() =>
            {
                shaderOverlayLoader.ResetShader();
                CloseMenu();
            });
        }

        // "None" = raw passthrough: show the live camera feed directly (already live).
        if (enableLivePreviews)
        {
            AssignPreview(noneTile, shaderOverlayLoader.GetCurrentCameraFeed());
        }

        noneTile.name = "ShaderTile_None";
    }

    /// <summary>
    /// Creates a tile for each shader in history.
    /// </summary>
    private void CreateShaderTiles()
    {
        int count = shaderOverlayLoader.GetShaderCount();

        for (int i = 0; i < count; i++)
        {
            string shaderName = shaderOverlayLoader.GetShaderName(i);
            string displayName = shaderOverlayLoader.GetShaderDisplayName(i);
            ParseDisplayName(displayName, out _, out string timestamp);

            CreateShaderTile(i, shaderName, timestamp);
        }
    }

    /// <summary>
    /// Creates a single shader tile in the menu.
    /// </summary>
    private void CreateShaderTile(int index, string shaderName, string timestamp)
    {
        var tile = Instantiate(tilePrefab, contentParent);
        var columns = tile.GetComponentsInChildren<TMPro.TextMeshProUGUI>();

        if (columns.Length >= 1) columns[0].text = ExtractShortShaderName(shaderName);
        if (columns.Length >= 2) columns[1].text = timestamp;

        if (tile.TryGetComponent<Button>(out var button))
        {
            int idx = index;
            // Ray / poke / trigger click applies the shader and closes the menu.
            button.onClick.AddListener(() =>
            {
                shaderOverlayLoader.SwitchShader(idx);
                CloseMenu();
            });
        }

        // Live preview: blit the camera feed through this shader's material into a
        // per-tile RenderTexture (slot index == shader history index).
        if (enableLivePreviews && previewRenderer != null)
        {
            var material = shaderOverlayLoader.GetShaderMaterial(index);
            var previewRT = previewRenderer.EnsureSlot(index, material);
            AssignPreview(tile, previewRT);
        }

        tile.name = $"ShaderTile_{index}";
    }

    /// <summary>
    /// Extracts short name from full shader path.
    /// </summary>
    private string ExtractShortShaderName(string shaderName)
    {
        if (string.IsNullOrEmpty(shaderName)) return "Unknown";
        
        if (shaderName.Contains("/"))
        {
            return shaderName[(shaderName.LastIndexOf('/') + 1)..];
        }
        
        return shaderName;
    }


    /// <summary>
    /// Parses display name to extract shader name and timestamp.
    /// </summary>
    private void ParseDisplayName(string displayName, out string name, out string timestamp)
    {
        var match = Regex.Match(displayName, ParsePattern, RegexOptions.Compiled);
        name = match.Groups[1].Value;
        
        if (match.Groups[2].Success)
        {
            string raw = match.Groups[2].Value;
            if (System.DateTime.TryParseExact(raw, TimestampFormat, null, System.Globalization.DateTimeStyles.None, out var dt))
            {
                timestamp = dt.ToString(DisplayDateFormat);
            }
            else
            {
                timestamp = raw;
            }
        }
        else
        {
            timestamp = "Current";
        }
    }

    /// <summary>
    /// Refreshes the menu content.
    /// </summary>
    public void RefreshMenu() => PopulateMenu();

    /// <summary>
    /// Toggles the menu visibility.
    /// </summary>
    public void ToggleMenu()
    {
        bool newState = !shaderHistoryMenu.activeSelf;
        
        if (newState)
        {
            // Only one panel is ever up.
            Menus.CloseOthers(this);

            shaderHistoryMenu.SetActive(true);
            selectedIndex = shaderOverlayLoader.GetCurrentShaderIndex() + 1;
            RefreshMenu();
        }
        else
        {
            shaderHistoryMenu.SetActive(false);
            OnMenuClosed();
        }
    }

    /// <summary>
    /// Closes the menu.
    /// </summary>
    public void CloseMenu()
    {
        if (shaderHistoryMenu != null && shaderHistoryMenu.activeSelf)
        {
            shaderHistoryMenu.SetActive(false);
            OnMenuClosed();
        }
    }

    /// <summary>
    /// Restores the detector vocabulary to the currently applied shader when the menu
    /// closes, undoing any browse-time vocabulary changes from the detection preview.
    /// </summary>
    private void OnMenuClosed()
    {
        vocabularyPushPending = false;
        vocabularyApplied = false;
        selectedDetectionsActive = false;
        if (!previewSelectedWithDetections) return;

        var provider = PreviewDetectionProvider;
        if (provider == null) return;

        if (TryLoadManifest(shaderOverlayLoader.GetActiveShaderFolder(), out var manifest))
        {
            provider.SetVocabulary(manifest.classes, manifest.confidence, manifest.cocoClasses);
        }
        else
        {
            provider.SetVocabulary(null);
        }
    }

    /// <summary>
    /// Binds the live detections to the selected tile's preview material (callback for the
    /// preview renderer). No-op when the detection preview is disabled.
    /// </summary>
    private void BindSelectedDetections(Material material)
    {
        // Only bind once the selected tile's vocabulary is active and settled; until then
        // the tile stays passthrough so stale detections never appear (kills the lag chain).
        if (previewSelectedWithDetections && selectedDetectionsActive && shaderOverlayLoader != null)
        {
            shaderOverlayLoader.BindCurrentDetectionsToPreview(material);
        }
    }

    /// <summary>
    /// Activates the selected shader's open-vocabulary classes on the detector, so the
    /// selected tile previews real detections. "None" (or a shader without a manifest)
    /// clears the vocabulary.
    /// </summary>
    private void PushSelectedVocabulary()
    {
        var provider = PreviewDetectionProvider;
        if (provider == null) return;

        int shaderIndex = selectedIndex - 1; // < 0 for the "None" tile
        if (shaderIndex >= 0
            && TryLoadManifest(shaderOverlayLoader.GetShaderFolder(shaderIndex), out var manifest))
        {
            provider.SetVocabulary(manifest.classes, manifest.confidence, manifest.cocoClasses);
        }
        else
        {
            provider.SetVocabulary(null);
        }
    }

    private static bool TryLoadManifest(string folder, out ShaderDetectionManifest manifest)
    {
        manifest = null;
        return !string.IsNullOrEmpty(folder)
            && ShaderDetectionManifest.TryLoad(folder, out manifest)
            && manifest.HasClasses;
    }

    /// <summary>
    /// Gets whether the menu is currently open.
    /// </summary>
    public bool IsMenuOpen => shaderHistoryMenu != null && shaderHistoryMenu.activeSelf;

    /// <summary>
    /// Dims non-selected tiles and highlights the selected one (preview tint + scale +
    /// background), points the preview renderer at the selected slot, and schedules its
    /// vocabulary activation. Scrolls the selection into view.
    /// </summary>
    private void UpdateTileHighlight()
    {
        int count = GetSelectableCount();
        selectedIndex = Mathf.Clamp(selectedIndex, 0, Mathf.Max(0, count - 1));

        for (int i = 0; i < contentParent.childCount && i < count; i++)
        {
            var child = contentParent.GetChild(i);
            bool isSelected = (i == selectedIndex);

            // Dim the preview itself (visible even behind an opaque tile background).
            var raw = child.GetComponentInChildren<RawImage>(true);
            if (raw != null)
            {
                raw.color = isSelected ? selectedTint : unselectedTint;
            }

            // Optional background highlight + a subtle scale bump on the selected tile.
            if (child.TryGetComponent<Image>(out var bg))
            {
                bg.color = isSelected ? HighlightColor : DefaultColor;
            }
            child.localScale = isSelected ? Vector3.one * selectedScale : Vector3.one;
        }

        // Selected tile = shader slot (selectedIndex 0 is the "None" passthrough tile).
        int selectedSlot = (selectedIndex > 0) ? selectedIndex - 1 : -1;
        previewRenderer?.SetSelectedSlot(selectedSlot);

        // Reset the detection-preview state for the new selection: defer the vocabulary
        // push (debounced) and keep detections off until it has settled (no stale effects).
        if (previewSelectedWithDetections)
        {
            selectionChangedTime = Time.unscaledTime;
            vocabularyPushPending = true;
            vocabularyApplied = false;
            selectedDetectionsActive = false;
        }

        ScrollToSelectedTile();
    }

    private static readonly Vector3[] CornerBuffer = new Vector3[4];

    /// <summary>
    /// Keeps the selected tile fully visible. The first/last row snaps the scroll fully to
    /// the top/bottom (so those rows are never clipped against the viewport edge); middle
    /// rows scroll only as much as needed. Independent of the content pivot; forces a layout
    /// update first so positions are current right after a selection change.
    /// </summary>
    private void ScrollToSelectedTile()
    {
        if (scrollRect == null || !scrollRect.vertical) return;
        if (contentParent == null || selectedIndex < 0 || selectedIndex >= contentParent.childCount) return;

        int cols = GetColumnCount();
        int count = GetSelectableCount();
        if (count <= 0) return;
        int row = selectedIndex / cols;
        int lastRow = (count - 1) / cols;

        Canvas.ForceUpdateCanvases();

        // Edge rows: snap fully so they are shown completely (1 = top, 0 = bottom).
        if (row <= 0) { scrollRect.verticalNormalizedPosition = 1f; return; }
        if (row >= lastRow) { scrollRect.verticalNormalizedPosition = 0f; return; }

        // Middle rows: scroll only enough to bring the tile into view (with a small margin).
        var target = contentParent.GetChild(selectedIndex) as RectTransform;
        var viewport = scrollRect.viewport != null ? scrollRect.viewport : (RectTransform)scrollRect.transform;
        if (target == null || viewport == null) return;

        float scrollable = contentParent.rect.height - viewport.rect.height;
        if (scrollable <= 1f) return; // everything fits — nothing to scroll

        // Tile and viewport edges in the viewport's local space (pivot-correct).
        target.GetWorldCorners(CornerBuffer);
        float tileTop = viewport.InverseTransformPoint(CornerBuffer[1]).y;     // top-left
        float tileBottom = viewport.InverseTransformPoint(CornerBuffer[0]).y;  // bottom-left
        float viewTop = viewport.rect.yMax;
        float viewBottom = viewport.rect.yMin;
        float margin = Mathf.Min(viewport.rect.height * 0.08f, target.rect.height * 0.25f);

        float move = 0f;
        if (tileTop > viewTop - margin) move = tileTop - (viewTop - margin);             // above view → scroll up
        else if (tileBottom < viewBottom + margin) move = tileBottom - (viewBottom + margin); // below view → scroll down

        if (Mathf.Abs(move) < 0.5f) return; // already fully visible

        float delta = move / scrollable;
        scrollRect.verticalNormalizedPosition =
            Mathf.Clamp01(scrollRect.verticalNormalizedPosition + delta);
    }

    /// <summary>
    /// Handles bundle received event.
    /// </summary>
    private void OnBundleReceived()
    {
        RefreshMenu();
    }
}
