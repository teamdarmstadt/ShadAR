using TMPro;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

/// <summary>
/// Shared look and uGUI builders for the panels (settings, shader history), so every panel
/// reads as one interface. Pull in with <c>using static UiKit;</c>.
/// </summary>
public static class UiKit
{
    // Near-opaque plate: a translucent one lets the differently converged video bleed through.
    public static readonly Color PanelColor = new(0.05f, 0.055f, 0.07f, 0.94f);
    public static readonly Color RowColor = new(0.16f, 0.18f, 0.22f, 0.92f);
    public static readonly Color RowHighlightColor = new(0.26f, 0.34f, 0.46f, 0.95f);
    public static readonly Color AccentColor = new(0.3f, 0.6f, 1f, 1f);
    public static readonly Color CaptionColor = new(1f, 1f, 1f, 0.45f);
    public static readonly Color ValueColor = new(0.55f, 0.75f, 1f, 1f);

    public static ColorBlock RowColors(Color normal) => new()
    {
        normalColor = normal,
        highlightedColor = normal == AccentColor ? AccentColor : RowHighlightColor,
        pressedColor = AccentColor,
        selectedColor = normal, // keeps the row from staying lit after a ray click
        disabledColor = new Color(normal.r, normal.g, normal.b, normal.a * 0.4f),
        colorMultiplier = 1f,
        fadeDuration = 0.06f,
    };

    public static RectTransform NewRect(string name, Transform parent)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.layer = parent.gameObject.layer;
        var rect = (RectTransform)go.transform;
        rect.SetParent(parent, false);
        return rect;
    }

    /// <summary>A row-coloured button. <paramref name="onClick"/> may be null (hold-style buttons).</summary>
    public static Button Button(Transform parent, string name, Color normal, UnityAction onClick)
    {
        var rect = NewRect(name, parent);
        var image = rect.gameObject.AddComponent<Image>();
        image.color = Color.white; // tinted by the ColorBlock
        var button = rect.gameObject.AddComponent<Button>();
        button.targetGraphic = image;
        button.colors = RowColors(normal);
        if (onClick != null) button.onClick.AddListener(onClick);
        return button;
    }

    public static TextMeshProUGUI Label(Transform parent, string name, float fontSize, TextAlignmentOptions alignment)
    {
        var rect = NewRect(name, parent);
        var label = rect.gameObject.AddComponent<TextMeshProUGUI>();
        label.fontSize = fontSize;
        label.color = Color.white;
        label.alignment = alignment;
        label.textWrappingMode = TextWrappingModes.NoWrap;
        label.overflowMode = TextOverflowModes.Ellipsis;
        label.raycastTarget = false; // clicks belong to the owning button
        return label;
    }

    public static T Ensure<T>(GameObject go) where T : Component =>
        go.TryGetComponent(out T existing) ? existing : go.AddComponent<T>();

    public static void Size(RectTransform rect, float height)
    {
        var element = Ensure<LayoutElement>(rect.gameObject);
        element.minHeight = element.preferredHeight = height;
    }

    public static void Stretch(RectTransform rect, float horizontalPadding = 0f)
    {
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = new Vector2(horizontalPadding, 0f);
        rect.offsetMax = new Vector2(-horizontalPadding, 0f);
    }
}
