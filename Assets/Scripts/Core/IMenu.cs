using UnityEngine;

/// <summary>
/// A head-locked panel that is opened and closed by the user. Only one is up at a time:
/// a menu calls <see cref="Menus.CloseOthers"/> when it opens.
/// </summary>
public interface IMenu
{
    bool IsMenuOpen { get; }
    void CloseMenu();
}

public static class Menus
{
    /// <summary>Closes every open menu except <paramref name="keep"/> (null = close all).</summary>
    public static void CloseOthers(IMenu keep)
    {
        // ponytail: scene scan on open/close only (a user action), not per frame.
        foreach (var behaviour in Object.FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None))
        {
            if (behaviour is IMenu menu && !ReferenceEquals(menu, keep) && menu.IsMenuOpen) menu.CloseMenu();
        }
    }
}
