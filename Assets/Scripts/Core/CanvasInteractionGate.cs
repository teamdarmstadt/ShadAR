using UnityEngine;

/// <summary>
/// Enables the ISDK canvas ray/poke interactables only while some interactive UI is actually
/// visible — a menu panel is open, or the reference bar is showing. The ISDK canvas interactable
/// spans the whole (always-active) UI canvas, so without this gate the controller ray would draw
/// against that invisible surface even with every menu closed (it hovers "nothing you can see").
///
/// Pair this with <c>ControllerRayVisual._hideWhenNoInteractable = true</c>: the ray then only
/// appears while it is over an ENABLED interactable, i.e. only when a menu/bar is up.
/// </summary>
public class CanvasInteractionGate : MonoBehaviour
{
    [Tooltip("Panels whose visibility means 'there is something to point at' — the menu panels " +
             "and the reference bar. The gate opens when ANY of them is active in the hierarchy.")]
    [SerializeField] private GameObject[] uiPanels;

    [Tooltip("The ISDK canvas-interaction objects (RayInteractable/PokeInteractable + PointableCanvas) " +
             "to enable/disable in step with the panels above.")]
    [SerializeField] private GameObject[] interactionObjects;

    private void OnEnable() => Apply(AnyPanelVisible());

    private void Update() => Apply(AnyPanelVisible());

    private bool AnyPanelVisible()
    {
        if (uiPanels == null) return false;

        foreach (var panel in uiPanels)
        {
            if (panel != null && panel.activeInHierarchy) return true;
        }
        return false;
    }

    private void Apply(bool open)
    {
        if (interactionObjects == null) return;
        foreach (var go in interactionObjects)
        {
            if (go != null && go.activeSelf != open) go.SetActive(open);
        }
    }
}
