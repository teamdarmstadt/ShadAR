using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;

/// <summary>
/// Loads the compiled shader bundles from the shader-history folders and keeps them as the
/// shader history (oldest first).
/// </summary>
public class ShaderBundleManager
{
    private const string BundleFileName = "shader.bundle";
    private const string SessionFolderPattern = "shader_*";
    private static readonly Regex SessionFolderRegex = new(@"^shader_\d{8}_\d{6}$");

    private readonly Logger logger;

    /// <summary>
    /// The loaded shaders with their metadata.
    /// </summary>
    public List<(string bundlePath, string displayName, string shaderName, Material material)> ShaderHistory { get; } = new();

    /// <summary>
    /// Index of the active shader, -1 for none (plain passthrough).
    /// </summary>
    public int CurrentShaderIndex { get; private set; } = -1;

    public ShaderBundleManager(Logger logger)
    {
        this.logger = logger;
    }

    /// <summary>
    /// Scans the history folder for bundles and rebuilds the history. Bundles that are already
    /// loaded are reused, so re-opening the menu only loads what is new.
    /// </summary>
    /// <param name="onLatestShaderApplied">When set, the newest shader becomes active and is passed
    /// here. When null (plain refresh) the previously active shader stays selected.</param>
    public IEnumerator ScanAndLoad(Action<Material> onLatestShaderApplied)
    {
        var dir = ShaderHistorySession.GetHistoryBaseFolderPath();
        if (!Directory.Exists(dir))
        {
            yield break;
        }

        var sessionFolders = Directory.GetDirectories(dir, SessionFolderPattern)
            .Where(d => SessionFolderRegex.IsMatch(Path.GetFileName(d)))
            .Where(d => File.Exists(Path.Combine(d, BundleFileName)))
            .OrderBy(d => Path.GetFileName(d))
            .ToList();

        // Identity is the bundle path; the index is unstable across a scan.
        string prevBundlePath = GetCurrentBundlePath();
        var loaded = ShaderHistory.ToDictionary(s => s.bundlePath);

        ShaderHistory.Clear();
        foreach (var folder in sessionFolders)
        {
            var bundlePath = Path.Combine(folder, BundleFileName);
            if (loaded.TryGetValue(bundlePath, out var entry))
            {
                ShaderHistory.Add(entry);
                continue;
            }

            yield return LoadAndStore(bundlePath, Path.GetFileName(folder));
        }

        if (onLatestShaderApplied == null)
        {
            CurrentShaderIndex = string.IsNullOrEmpty(prevBundlePath)
                ? -1
                : ShaderHistory.FindIndex(s => s.bundlePath == prevBundlePath);
        }
        else
        {
            CurrentShaderIndex = ShaderHistory.Count - 1;
            if (CurrentShaderIndex >= 0)
            {
                onLatestShaderApplied(ShaderHistory[CurrentShaderIndex].material);
                logger?.LogInfo($"Applied latest shader: {ShaderHistory[CurrentShaderIndex].displayName}", nameof(ShaderBundleManager));
                logger?.SetStatus("Shader applied.");
            }
        }
    }

    private IEnumerator LoadAndStore(string bundlePath, string displayName)
    {
        var bundle = AssetBundle.LoadFromFile(bundlePath);
        if (bundle == null)
        {
            // The two real causes are a bundle built by a different Unity version or for a
            // different build target; the bundle header names the building editor.
            logger?.LogError(
                $"Bundle load failed: {DescribeBundle(bundlePath)}. This app runs Unity " +
                $"{Application.unityVersion} / Android. Rebuild it on a QuestPackager using the " +
                "same Unity version with the Android module installed.",
                nameof(ShaderBundleManager));
            yield break;
        }

        var shaderPath = bundle.GetAllAssetNames().FirstOrDefault(a => a.EndsWith(".shader"));
        var shader = shaderPath != null ? bundle.LoadAsset<Shader>(shaderPath) : null;
        if (shader == null)
        {
            logger?.LogError($"No shader in bundle {displayName}.", nameof(ShaderBundleManager));
            bundle.Unload(false);
            yield break;
        }

        var mat = new Material(shader) { hideFlags = HideFlags.DontSave };
        ShaderHistory.Add((bundlePath, displayName, shader.name, mat));

        bundle.Unload(false);
        yield return null;
    }

    /// <summary>
    /// Switches to a shader from history by index.
    /// </summary>
    public void SwitchShader(int index, Action<Material> onShaderSwitched)
    {
        if (index < 0 || index >= ShaderHistory.Count)
        {
            logger?.LogError($"Invalid shader index: {index}", nameof(ShaderBundleManager));
            return;
        }

        CurrentShaderIndex = index;
        onShaderSwitched?.Invoke(ShaderHistory[index].material);
        logger?.LogInfo($"Switched to shader {ShaderHistory[index].displayName}.", nameof(ShaderBundleManager));
    }

    /// <summary>
    /// Marks "no shader" (plain passthrough) as active.
    /// </summary>
    public void ClearSelection() => CurrentShaderIndex = -1;

    public string GetCurrentBundlePath() => GetBundlePath(CurrentShaderIndex);

    public Material GetShaderMaterial(int index) =>
        (index >= 0 && index < ShaderHistory.Count) ? ShaderHistory[index].material : null;

    public string GetBundlePath(int index) =>
        (index >= 0 && index < ShaderHistory.Count) ? ShaderHistory[index].bundlePath : null;

    public int GetShaderCount() => ShaderHistory.Count;

    public int GetCurrentShaderIndex() => CurrentShaderIndex;

    public string GetShaderDisplayName(int index) =>
        (index >= 0 && index < ShaderHistory.Count) ? ShaderHistory[index].displayName : null;

    public string GetShaderName(int index) =>
        (index >= 0 && index < ShaderHistory.Count) ? ShaderHistory[index].shaderName : null;

    /// <summary>
    /// Size plus the Unity version that BUILT a bundle, read from its header.
    /// Layout: "UnityFS\0" then a format int, then NUL-terminated "5.x.x" and the real editor
    /// version ("6000.3.11f1"). Never throws — this only ever runs on an already-failed load.
    /// </summary>
    private static string DescribeBundle(string bundlePath)
    {
        try
        {
            long size = new FileInfo(bundlePath).Length;
            if (size == 0)
            {
                return "0 bytes (the compile server sent nothing back)";
            }

            var head = new byte[48];
            int read;
            using (var fs = File.OpenRead(bundlePath))
            {
                read = fs.Read(head, 0, head.Length);
            }

            string built = System.Text.Encoding.ASCII.GetString(head, 0, Mathf.Max(read, 0))
                .Split('\0')
                .LastOrDefault(t => t.Length > 2 && char.IsDigit(t[0]) && t.Contains('.'));

            return $"{size} bytes, built with Unity {built ?? "unknown"}";
        }
        catch (Exception ex)
        {
            return $"unreadable ({ex.Message})";
        }
    }
}
