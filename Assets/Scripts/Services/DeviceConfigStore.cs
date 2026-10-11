using System;
using System.IO;
using UnityEngine;

/// <summary>
/// Shared device-side JSON config I/O for the *ConfigManager singletons.
/// Resolves a path under persistentDataPath, writes an empty template when the
/// file is missing, reads it back, and merges values against fallbacks.
/// </summary>
public static class DeviceConfigStore
{
    public static string ResolvePath(string fileName) =>
        Path.Combine(Application.persistentDataPath, fileName);

    /// <summary>
    /// Loads <typeparamref name="T"/> from <paramref name="path"/>, writing
    /// <paramref name="template"/> (or an empty instance) there first if the file does not
    /// exist. Never returns null.
    /// </summary>
    public static T LoadOrCreate<T>(string path, Logger logger, string context, T template = null) where T : class, new()
    {
        template ??= new T();
        try
        {
            if (!File.Exists(path))
            {
                File.WriteAllText(path, JsonUtility.ToJson(template, true));
                logger?.LogInfo($"Created config template at: {path} (edit it on the device).", context);
            }

            return JsonUtility.FromJson<T>(File.ReadAllText(path)) ?? template;
        }
        catch (Exception ex)
        {
            logger?.LogError($"Device config load failed ({path}): {ex.Message}", context);
            return template;
        }
    }

    /// <summary>Primary value if set, else the fallback.</summary>
    public static string Merge(string primary, string fallback) =>
        string.IsNullOrWhiteSpace(primary) ? fallback : primary;

    /// <summary>Primary value if positive, else the fallback.</summary>
    public static int Merge(int primary, int fallback) =>
        primary > 0 ? primary : fallback;
}
