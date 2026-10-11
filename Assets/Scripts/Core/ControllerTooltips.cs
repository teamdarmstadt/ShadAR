using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

/// <summary>
/// Floating button hints attached to the Touch controllers: a dark label plate joined by a
/// leader line to a bright pip sitting on the control it describes.
///
/// The hints are not hard-wired: whoever owns a button registers it with
/// <see cref="Register"/> — the settings panel its Y, the shader history its X, push-to-talk
/// its B. A hint tied to an
/// <see cref="IMenu"/> reads "Close UI" while that menu is open, because that is what the next
/// press does; reading IsMenuOpen (instead of latching on the press) also corrects the label when
/// a menu closes on its own. The right trigger's "Button Click" (ray select) is registered here.
///
/// Self-contained: drop this on any GameObject and it finds the rig itself, then builds each
/// tooltip at runtime the first time a hint for its button shows up. No prefabs.
///
/// Built on a per-tooltip **world-space Canvas** with uGUI `Image` + `TextMeshProUGUI`. That is
/// a deliberate choice over 3D meshes: under Single Pass Instanced a shader without the stereo
/// instancing macros renders into ONE eye only, and `UI/Default` + TMP's UGUI shaders are the
/// ones this project already proves stereo-correct on its other canvases. It also supplies the
/// plate and line materials without a runtime `Shader.Find`, which could miss in a build.
///
/// The whole assembly is 2D inside a canvas centred ON the control and billboarded to the eye,
/// so the leader line needs no 3D math and always reads correctly from the viewer's angle. Line,
/// pip and plate geometry are computed once at build time; per frame this costs one transform
/// write plus one look-rotation per visible tooltip. Text is assigned only when the string
/// actually changes — TMP re-meshes on every set and would otherwise rebuild the canvas per frame.
/// </summary>
public class ControllerTooltips : MonoBehaviour
{
    /// <summary>One registered hint. Keep the handle to <see cref="Unregister"/> it.</summary>
    public sealed class Hint
    {
        public OVRInput.Button Button;
        public string Label;
        public string ModeName; // null = every mode
        public IMenu Menu;      // optional: label flips to "Close UI" while it is open
    }

    private static readonly List<Hint> hints = new();

    /// <summary>
    /// Shows <paramref name="label"/> on <paramref name="button"/>. Several hints may share a
    /// button; the first one whose <paramref name="modeName"/> matches the active mode (or is
    /// null) is shown. Safe to call before this component exists.
    /// </summary>
    public static Hint Register(OVRInput.Button button, string label, string modeName = null, IMenu menu = null)
    {
        var hint = new Hint { Button = button, Label = label, ModeName = modeName, Menu = menu };
        hints.Add(hint);
        return hint;
    }

    public static void Unregister(Hint hint) => hints.Remove(hint);

    /// <summary>Where a control sits on its controller and which way its label points.</summary>
    private readonly struct ControlGeometry
    {
        public readonly bool Left;
        public readonly string[] Bones;
        public readonly Vector3 FallbackPoint; // controller-local metres, used when no bone resolves
        public readonly Vector2 LabelOffset;   // canvas units; the leader line spans exactly this

        public ControlGeometry(bool left, string[] bones, Vector3 fallbackPoint, Vector2 labelOffset)
        {
            Left = left;
            Bones = bones;
            FallbackPoint = fallbackPoint;
            LabelOffset = labelOffset;
        }
    }

    // ponytail: fixed table; buttons on the same controller get offsets pointing different ways
    // so their plates never stack. Add a row when a mode needs another control.
    private static readonly Dictionary<OVRInput.Button, ControlGeometry> geometry = new()
    {
        [OVRInput.Button.Three] = new(true, new[] { "b_button_x" }, new(0.005f, 0.005f, 0.015f), new(150f, 85f)),
        [OVRInput.Button.Four] = new(true, new[] { "b_button_y" }, new(0.005f, 0.005f, 0.028f), new(-150f, 120f)),
        [OVRInput.Button.One] = new(false, new[] { "b_button_a" }, new(-0.005f, 0.005f, 0.015f), new(-150f, 60f)),
        [OVRInput.Button.Two] = new(false, new[] { "b_button_b" }, new(-0.005f, 0.005f, 0.028f), new(-150f, 120f)),
        [OVRInput.Button.PrimaryIndexTrigger] = new(true, new[] { "b_trigger_front", "left_b_trigger_front" }, new(0f, -0.015f, 0.03f), new(-150f, 85f)),
        [OVRInput.Button.SecondaryIndexTrigger] = new(false, new[] { "b_trigger_front", "right_b_trigger_front" }, new(0f, -0.015f, 0.03f), new(150f, 85f)),
        [OVRInput.Button.PrimaryThumbstick] = new(true, new[] { "b_thumbstick", "left_b_thumbstick" }, new(0f, 0.012f, 0.005f), new(150f, 160f)),
        [OVRInput.Button.SecondaryThumbstick] = new(false, new[] { "b_thumbstick", "right_b_thumbstick" }, new(0f, 0.012f, 0.005f), new(150f, 160f)),
    };

    [Header("Rig (auto-found when empty)")]
    [SerializeField] private OVRCameraRig cameraRig;

    [Header("Label text")]
    [SerializeField] private string closeUiText = "Close UI";
    [Tooltip("Hint on the right trigger, which selects whatever the controller ray points at.")]
    [SerializeField] private string triggerText = "Select";

    [Serializable]
    private struct ManualAnchor
    {
        public OVRInput.Button button;
        public Transform anchor;
    }

    [Header("Pip anchoring — manual (highest priority)")]
    [Tooltip("Optional hand-placed anchors. Assign an empty GameObject sitting ON the control and " +
             "the pip goes exactly there, overriding the bone lookup below. This is the " +
             "what-you-see-is-what-you-get route: parent the empty to LeftControllerAnchor / " +
             "RightControllerAnchor, temporarily enable the controller model to see the button, " +
             "drag the empty onto it, then disable the model again. Parent it to the ANCHOR " +
             "rather than the model so the placement survives a controller-generation swap. " +
             "Select this component to see a gizmo at each pip position while you align them.")]
    [SerializeField] private List<ManualAnchor> manualAnchors = new();

    [Header("Pip anchoring — automatic, via the model's button bones")]
    [Tooltip("Anchor each pip to the controller model's named button bone instead of a " +
             "hand-tuned offset. This is exact for whichever controller generation is " +
             "connected and needs no tuning. Bones resolve even while the model is hidden: " +
             "Transform world positions are valid for inactive objects, and the parent " +
             "(LeftControllerAnchor) is posed by OVR every frame.")]
    [SerializeField] private bool anchorToButtonBones = true;
    [Tooltip("Every controller generation's model is present under the anchor and only one is " +
             "shown at a time. A VISIBLE bone always wins; when none is visible (the controller " +
             "model building blocks are disabled in this scene) the first generation listed here " +
             "that is present wins. Quest 3 ships Touch Plus, hence the order.")]
    [SerializeField] private string[] preferredModelRoots =
        { "MetaQuestTouchPlus", "MetaQuestTouchPro", "OculusTouchForQuest2", "OculusTouchForQuestAndRiftS" };
    [Tooltip("Seconds between re-resolve attempts while a bone is missing, and re-checks that " +
             "the cached bone is still the best candidate after a controller/model swap.")]
    [SerializeField] private float boneResolveInterval = 1f;
    [Tooltip("Moves the pip this far toward the eye so it is not buried inside the controller " +
             "mesh. Free of parallax: shifting along the view ray leaves the projected " +
             "position identical, it only wins the depth test.")]
    [SerializeField] private float pipTowardViewer = 0.01f;

    [Header("Layout (canvas units; 1 unit = canvasScale metres)")]
    [SerializeField] private float canvasScale = 0.0004f;
    [SerializeField] private float fontSize = 30f;
    [SerializeField] private float platePadding = 14f;
    [SerializeField] private float lineThickness = 3f;
    [SerializeField] private float pipSize = 12f;

    [Header("Colours")]
    [Tooltip("Kept fairly dark: a very translucent plate lets the differently-converged " +
             "passthrough video bleed through the glyphs and causes binocular rivalry.")]
    [SerializeField] private Color plateColor = new(0f, 0f, 0f, 0.6f);
    [SerializeField] private Color textColor = new(1f, 1f, 1f, 0.9f);
    [SerializeField] private Color lineColor = new(1f, 1f, 1f, 0.5f);
    [Tooltip("The pip marking the control. Warm so it stands out against the black controller shells.")]
    [SerializeField] private Color pipColor = new(1f, 0.85f, 0.2f, 0.85f);

    [Header("Visibility")]
    [Tooltip("Master switch; also settable at runtime via SetVisible().")]
    [SerializeField] private bool showTooltips = true;
    [Tooltip("A controller's tooltips only appear while you look at it: the angle between the " +
             "view direction and the controller must be below this. 0 = always shown.")]
    [SerializeField, Range(0f, 90f)] private float gazeConeDegrees = 22f;
    [Tooltip("Extra degrees before a shown tooltip hides again, so it does not flicker at the edge.")]
    [SerializeField] private float gazeHysteresisDegrees = 8f;
    [SerializeField] private float fadeSeconds = 0.15f;
    [Tooltip("Hide a controller's tooltips while it is not tracked, so they vanish when the " +
             "controllers are put down for hand tracking.")]
    [SerializeField] private bool hideWhenUntracked = true;
    [Tooltip("Layer for the tooltip canvases. MUST be in the centre camera's culling mask — " +
             "5 (UI) is the layer this project's other canvases already use.")]
    [SerializeField] private int tooltipLayer = 5;

    [Header("Controller model brightness")]
    [Tooltip("The scene has no lights, and the controller models use Meta/Lit (URP ForwardLit), " +
             "so they render almost black. This raises the scene's flat ambient, which is the " +
             "only thing lighting them. Free at runtime — no light, no extra pass. Everything " +
             "else in this app is unlit, so nothing else changes.")]
    [SerializeField] private bool brightenControllerModels = true;
    [SerializeField] private Color ambientLight = new(0.78f, 0.78f, 0.8f, 1f);

    private Transform leftAnchor;
    private Transform rightAnchor;
    private Transform headAnchor;
    private Hint triggerHint;
    private bool leftLooked;
    private bool rightLooked;

    // Own root, deliberately NOT parented to this component's object: the natural host is the
    // `UI` canvas, whose transform is scaled to ~0.0006 to give the canvas its angular size.
    // Inheriting that would shrink the tooltips to nothing, so they live at the scene root and
    // are driven entirely by world-space writes.
    private Transform tooltipRoot;

    // One per button that has ever had a hint; built lazily, reused across modes.
    private readonly List<Tooltip> tooltips = new();

    /// <summary>One tooltip: the billboarded canvas plus the label whose text can change.</summary>
    private class Tooltip
    {
        public OVRInput.Button Button;
        public ControlGeometry Geometry;
        public Transform Root;
        public CanvasGroup Group;
        public TextMeshProUGUI Label;
        public string CurrentText;

        // Hand-placed anchor from the Inspector. Wins over everything when assigned.
        public Transform ManualAnchor;

        // Exact anchor on the controller model, resolved by bone name. Null until resolved (or
        // permanently, if this controller's model does not carry a listed bone) — the fallback
        // offset takes over then.
        public Transform Bone;
        public float NextResolveTime;
    }

    private void Awake()
    {
        if (cameraRig == null) cameraRig = FindAnyObjectByType<OVRCameraRig>();
        if (cameraRig != null)
        {
            leftAnchor = cameraRig.leftControllerAnchor;
            rightAnchor = cameraRig.rightControllerAnchor;
            headAnchor = cameraRig.centerEyeAnchor;
        }

        tooltipRoot = new GameObject("ControllerTooltips").transform;
        triggerHint = Register(OVRInput.Button.SecondaryIndexTrigger, triggerText);
    }

    private void Start()
    {
        if (!brightenControllerModels)
        {
            return;
        }

        // Flat ambient is the whole lighting rig: the controller models are the only lit
        // surfaces in the scene, so this is cheaper and more predictable than adding a light.
        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = ambientLight;
    }

    private void OnDestroy()
    {
        Unregister(triggerHint);
        // The tooltip root sits outside this object's hierarchy, so it is not cleaned up with it.
        if (tooltipRoot != null)
        {
            Destroy(tooltipRoot.gameObject);
        }
    }

    /// <summary>Shows or hides every tooltip (e.g. to keep hints out of a study condition).</summary>
    public void SetVisible(bool visible) => showTooltips = visible;

    // -------------------------------------------------------------------------
    // Per-frame placement
    // -------------------------------------------------------------------------

    private void LateUpdate()
    {
        // LateUpdate, not Update: the controller anchors are posed during Update, so placing the
        // tooltips here means they never trail the controller by a frame.
        EnsureTooltipsForHints();

        leftLooked = LookedAt(leftAnchor, leftLooked);
        rightLooked = LookedAt(rightAnchor, rightLooked);
        bool leftVisible = showTooltips && leftLooked && IsControllerUsable(OVRInput.Controller.LTouch);
        bool rightVisible = showTooltips && rightLooked && IsControllerUsable(OVRInput.Controller.RTouch);
        const string mode = null; // ShadAR has a single mode, so every hint registers for all of them
        float fadeStep = fadeSeconds > 0f ? Time.unscaledDeltaTime / fadeSeconds : 1f;

        foreach (var tooltip in tooltips)
        {
            var hint = ActiveHint(tooltip.Button, mode);
            bool visible = hint != null && (tooltip.Geometry.Left ? leftVisible : rightVisible);
            if (!Fade(tooltip, visible, fadeStep)) continue;

            // Still fading out after its hint went away (mode switch): keep the last text.
            if (hint != null) SetText(tooltip, hint.Menu != null && hint.Menu.IsMenuOpen ? closeUiText : hint.Label);
            Place(tooltip, tooltip.Geometry.Left ? leftAnchor : rightAnchor);
        }
    }

    /// <summary>
    /// Whether the controller sits inside the gaze cone. Hysteresis: once shown, it stays shown
    /// until the controller leaves a slightly wider cone.
    /// </summary>
    private bool LookedAt(Transform anchor, bool wasLooked)
    {
        if (anchor == null) return false;
        if (gazeConeDegrees <= 0f || headAnchor == null) return true;
        float angle = Vector3.Angle(headAnchor.forward, anchor.position - headAnchor.position);
        return angle < gazeConeDegrees + (wasLooked ? gazeHysteresisDegrees : 0f);
    }

    private static Hint ActiveHint(OVRInput.Button button, string mode)
    {
        foreach (var hint in hints)
        {
            if (hint.Button == button && (hint.ModeName == null || hint.ModeName == mode)) return hint;
        }
        return null;
    }

    private void EnsureTooltipsForHints()
    {
        // Cheap when nothing is new: one pass over a handful of hints.
        foreach (var hint in hints)
        {
            if (HasTooltip(hint.Button)) continue;
            if (!geometry.TryGetValue(hint.Button, out var geo))
            {
                Debug.LogWarning($"{nameof(ControllerTooltips)}: no geometry for {hint.Button}; hint '{hint.Label}' is not shown.");
                geometry[hint.Button] = default; // warn once; Bones == null marks it unusable
                continue;
            }
            if (geo.Bones == null) continue;
            tooltips.Add(BuildTooltip(hint.Button, geo));
        }
    }

    private bool HasTooltip(OVRInput.Button button)
    {
        foreach (var tooltip in tooltips)
        {
            if (tooltip.Button == button) return true;
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    private Tooltip BuildTooltip(OVRInput.Button button, ControlGeometry geo)
    {
        Vector2 offset = geo.LabelOffset;

        // Root canvas, centred on the control it points at.
        var canvasObject = new GameObject($"Tooltip_{button}", typeof(RectTransform), typeof(Canvas), typeof(CanvasGroup));
        canvasObject.layer = tooltipLayer;
        var canvas = canvasObject.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        var canvasRect = (RectTransform)canvasObject.transform;
        canvasRect.SetParent(tooltipRoot, false);
        canvasRect.sizeDelta = Vector2.one;      // container only; children place themselves
        canvasRect.localScale = Vector3.one * canvasScale;

        // Leader line: a rectangle pinned at the pip, rotated toward the plate. Its pivot sits
        // on the left edge so length grows along +X and the rotation is a plain 2D angle.
        var line = CreateGraphic("Line", canvasRect, lineColor);
        line.rectTransform.pivot = new Vector2(0f, 0.5f);
        line.rectTransform.anchoredPosition = Vector2.zero;
        line.rectTransform.sizeDelta = new Vector2(offset.magnitude, lineThickness);
        line.rectTransform.localRotation =
            Quaternion.Euler(0f, 0f, Mathf.Atan2(offset.y, offset.x) * Mathf.Rad2Deg);

        // Pip: marks the control itself. Drawn after the line so it sits on top of its end.
        var pip = CreateGraphic("Pip", canvasRect, pipColor);
        pip.rectTransform.anchoredPosition = Vector2.zero;
        pip.rectTransform.sizeDelta = new Vector2(pipSize, pipSize);

        // Plate: auto-sizes to the text via layout group + fitter (native uGUI, no code). It grows
        // away from the line end, so a label on the left of its control grows to the left.
        var plate = CreateGraphic("Plate", canvasRect, plateColor);
        plate.rectTransform.pivot = new Vector2(offset.x < 0f ? 1f : 0f, 0.5f);
        plate.rectTransform.anchoredPosition = offset;

        var layout = plate.gameObject.AddComponent<HorizontalLayoutGroup>();
        layout.padding = new RectOffset(
            (int)platePadding, (int)platePadding, (int)(platePadding * 0.5f), (int)(platePadding * 0.5f));
        layout.childAlignment = TextAnchor.MiddleCenter;
        layout.childControlWidth = true;
        layout.childControlHeight = true;
        layout.childForceExpandWidth = false;
        layout.childForceExpandHeight = false;

        var fitter = plate.gameObject.AddComponent<ContentSizeFitter>();
        fitter.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        var labelObject = new GameObject("Text", typeof(RectTransform), typeof(TextMeshProUGUI));
        labelObject.layer = tooltipLayer;
        labelObject.transform.SetParent(plate.rectTransform, false);

        var label = labelObject.GetComponent<TextMeshProUGUI>();
        label.fontSize = fontSize;
        label.color = textColor;
        label.alignment = TextAlignmentOptions.Center;
        label.textWrappingMode = TextWrappingModes.NoWrap;
        label.raycastTarget = false;

        canvasObject.SetActive(false);
        return new Tooltip
        {
            Button = button,
            Geometry = geo,
            Root = canvasRect,
            Group = canvasObject.GetComponent<CanvasGroup>(),
            Label = label,
            ManualAnchor = ManualAnchorFor(button),
        };
    }

    private Transform ManualAnchorFor(OVRInput.Button button)
    {
        foreach (var manual in manualAnchors)
        {
            if (manual.button == button) return manual.anchor;
        }
        return null;
    }

    private Image CreateGraphic(string name, RectTransform parent, Color color)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(Image));
        go.layer = tooltipLayer;
        var rect = (RectTransform)go.transform;
        rect.SetParent(parent, false);
        rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 0.5f);

        var image = go.GetComponent<Image>();
        image.color = color;
        image.raycastTarget = false; // never a target for the ISDK ray/poke interactors
        return image;
    }

    // -------------------------------------------------------------------------
    // Button-bone resolution
    // -------------------------------------------------------------------------

    /// <summary>
    /// Finds the tooltip's button bone under its controller anchor, throttled to
    /// <see cref="boneResolveInterval"/>. Throttling matters: the search allocates and walks the
    /// whole controller subtree (five model generations live under each anchor), which is fine
    /// once a second but not per frame. Re-runs periodically even once resolved, so a controller
    /// or model swap re-picks the right generation.
    /// </summary>
    private void ResolveBoneIfDue(Tooltip tooltip, Transform searchRoot)
    {
        // A hand-placed anchor is authoritative — never spend the search on it.
        if (tooltip.ManualAnchor != null || !anchorToButtonBones || searchRoot == null)
        {
            tooltip.Bone = null;
            return;
        }
        if (tooltip.Bone != null && Time.unscaledTime < tooltip.NextResolveTime)
        {
            return;
        }

        tooltip.NextResolveTime = Time.unscaledTime + Mathf.Max(0.1f, boneResolveInterval);
        tooltip.Bone = FindBone(searchRoot, tooltip.Geometry.Bones);
    }

    private Transform FindBone(Transform searchRoot, string[] boneNames)
    {
        Transform best = null;
        int bestRank = int.MaxValue;
        // `true` includes inactive: the model building blocks are disabled in this scene, and an
        // inactive Transform still reports a correct world position through its posed parents.
        foreach (Transform candidate in searchRoot.GetComponentsInChildren<Transform>(true))
        {
            if (Array.IndexOf(boneNames, candidate.name) < 0) continue;
            int rank = RankBone(candidate);
            if (rank < bestRank)
            {
                bestRank = rank;
                best = candidate;
            }
        }
        return best;
    }

    /// <summary>
    /// Lower is better. A bone on the model actually being displayed always wins; otherwise the
    /// generation's position in <see cref="preferredModelRoots"/> decides.
    /// </summary>
    private int RankBone(Transform bone)
    {
        if (bone.gameObject.activeInHierarchy)
        {
            return 0;
        }

        for (Transform p = bone; p != null; p = p.parent)
        {
            for (int i = 0; i < preferredModelRoots.Length; i++)
            {
                if (p.name.StartsWith(preferredModelRoots[i], StringComparison.OrdinalIgnoreCase))
                {
                    return 1 + i;
                }
            }
        }
        return int.MaxValue - 1; // present but an unlisted generation — still better than nothing
    }

    private bool IsControllerUsable(OVRInput.Controller controller)
    {
        if (!OVRInput.IsControllerConnected(controller))
        {
            return false;
        }
        // Connected but untracked is the hands-in-use case: the controller is on the desk.
        return !hideWhenUntracked || OVRInput.GetControllerPositionTracked(controller);
    }

    /// <summary>
    /// Centres the tooltip canvas on the control and turns it to face the viewer. The position
    /// rides the controller; the rotation does not — text rigidly parented to the controller
    /// becomes unreadable the moment the wrist rotates, and the leader line only reads correctly
    /// while the canvas plane is square to the eye.
    /// </summary>
    private void Place(Tooltip tooltip, Transform anchor)
    {
        ResolveBoneIfDue(tooltip, anchor);
        Vector3 position = AnchorPosition(tooltip, anchor);

        if (headAnchor == null)
        {
            tooltip.Root.position = position;
            return;
        }

        Vector3 away = position - headAnchor.position;
        if (away.sqrMagnitude < 1e-6f)
        {
            tooltip.Root.position = position;
            return;
        }

        // Pull toward the eye so the pip clears the controller mesh. This slides along the view
        // ray, so the pip still projects onto exactly the same spot — it only wins the depth test.
        if (pipTowardViewer > 0f)
        {
            position -= away.normalized * pipTowardViewer;
        }

        tooltip.Root.position = position;
        // A world-space Canvas faces +Z, so pointing +Z away from the eye shows its front.
        tooltip.Root.rotation = Quaternion.LookRotation(away, Vector3.up);
    }

    /// <summary>
    /// Where the pip goes, in priority order: a hand-placed Inspector anchor, then the model's
    /// button bone, then the fallback offset from the controller anchor.
    /// </summary>
    private static Vector3 AnchorPosition(Tooltip tooltip, Transform controllerAnchor)
    {
        if (tooltip.ManualAnchor != null)
        {
            return tooltip.ManualAnchor.position;
        }
        if (tooltip.Bone != null)
        {
            return tooltip.Bone.position;
        }
        return controllerAnchor != null
            ? controllerAnchor.TransformPoint(tooltip.Geometry.FallbackPoint)
            : Vector3.zero;
    }

    private static void SetText(Tooltip tooltip, string text)
    {
        if (tooltip.CurrentText == text)
        {
            return; // TMP re-meshes and dirties the canvas on every set — only touch on change
        }
        tooltip.Label.text = text;
        tooltip.CurrentText = text;
    }

    /// <summary>Fades toward shown/hidden; returns whether the tooltip is (still) on screen.</summary>
    private static bool Fade(Tooltip tooltip, bool visible, float step)
    {
        tooltip.Group.alpha = Mathf.MoveTowards(tooltip.Group.alpha, visible ? 1f : 0f, step);
        bool onScreen = tooltip.Group.alpha > 0f;
        if (tooltip.Root.gameObject.activeSelf != onScreen) tooltip.Root.gameObject.SetActive(onScreen);
        return onScreen;
    }

#if UNITY_EDITOR
    /// <summary>
    /// Draws a marker wherever each pip will appear, so the manual anchors can be dragged onto
    /// the buttons in the Scene view without entering Play mode. Selected-only, which also keeps
    /// the (allocating) bone search out of the normal editor loop.
    /// </summary>
    private void OnDrawGizmosSelected()
    {
        var rig = cameraRig != null ? cameraRig : FindAnyObjectByType<OVRCameraRig>();
        if (rig == null)
        {
            return;
        }

        foreach (var pair in geometry)
        {
            if (pair.Value.Bones == null) continue;
            var controllerAnchor = pair.Value.Left ? rig.leftControllerAnchor : rig.rightControllerAnchor;
            DrawPipGizmo(ManualAnchorFor(pair.Key), pair.Value, controllerAnchor, pair.Key.ToString());
        }
    }

    private void DrawPipGizmo(Transform manual, ControlGeometry geo, Transform controllerAnchor, string label)
    {
        Vector3 position;
        string source;

        if (manual != null)
        {
            position = manual.position;
            source = "manual";
        }
        else
        {
            var bone = anchorToButtonBones && controllerAnchor != null ? FindBone(controllerAnchor, geo.Bones) : null;
            if (bone != null)
            {
                position = bone.position;
                source = $"bone {bone.name}";
            }
            else if (controllerAnchor != null)
            {
                position = controllerAnchor.TransformPoint(geo.FallbackPoint);
                source = "fallback offset";
            }
            else
            {
                return;
            }
        }

        Gizmos.color = pipColor;
        Gizmos.DrawSphere(position, 0.006f);
        Gizmos.DrawWireSphere(position, 0.012f);
        UnityEditor.Handles.Label(position, $"{label} ({source})");
    }
#endif
}
