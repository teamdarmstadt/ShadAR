using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Per-shader detection vocabulary. Stored next to the compiled shader bundle as
/// <c>detection_classes.json</c> so the open-vocabulary detector can be configured
/// with exactly the objects this shader was generated for.
///
/// The order of <see cref="classes"/> is significant: a detection's slot index
/// (the position of its class in this list) becomes the <c>labelID</c> consumed by
/// the generated shader (<c>det.labelID == 0</c> = first class, etc.). This replaces
/// the previous dependency on static COCO label IDs.
/// </summary>
[Serializable]
public class ShaderDetectionManifest
{
    private const string FileName = "detection_classes.json";

    /// <summary>
    /// Natural-language target classes, in slot order. Slot index = shader labelID.
    /// </summary>
    public List<string> classes = new();

    /// <summary>
    /// COCO class name per slot, SAME LENGTH and order as <see cref="classes"/>, written by
    /// the LLM at generation time. Empty string = this slot has no sensible COCO equivalent.
    ///
    /// Only used by the closed-vocabulary detector (yolo26-seg), which cannot be told what to
    /// look for and would otherwise have to guess the translation at runtime from the natural
    /// language phrase. The LLM is in a far better position to decide: it knows the intent of
    /// the shader, it sees all slots at once, and it can spot that two phrases would need the
    /// same COCO class — which the wire format cannot express, since a detection carries one
    /// slot only. Older shaders have no such list; the server falls back to its synonym
    /// heuristic for every slot missing here, so they keep working.
    /// </summary>
    public List<string> cocoClasses = new();

    /// <summary>
    /// Confidence threshold the detector should apply for this shader (0..1).
    /// </summary>
    public float confidence = 0.25f;

    /// <summary>
    /// True when at least one target class is present.
    /// </summary>
    public bool HasClasses => classes != null && classes.Count > 0;

    /// <summary>
    /// Returns the canonical manifest path inside a session/history folder.
    /// </summary>
    public static string GetPath(string sessionFolderPath)
    {
        return string.IsNullOrEmpty(sessionFolderPath)
            ? null
            : Path.Combine(sessionFolderPath, FileName);
    }

    /// <summary>
    /// Writes the manifest as JSON into the given session folder.
    /// </summary>
    public void Save(string sessionFolderPath)
    {
        string path = GetPath(sessionFolderPath);
        if (string.IsNullOrEmpty(path))
        {
            return;
        }

        Directory.CreateDirectory(sessionFolderPath);
        File.WriteAllText(path, JsonUtility.ToJson(this, true));
    }

    /// <summary>
    /// Attempts to load a manifest from a session/history folder.
    /// Returns false (and a null manifest) when no manifest file exists or parsing fails.
    /// </summary>
    public static bool TryLoad(string sessionFolderPath, out ShaderDetectionManifest manifest)
    {
        manifest = null;
        string path = GetPath(sessionFolderPath);
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            return false;
        }

        try
        {
            manifest = JsonUtility.FromJson<ShaderDetectionManifest>(File.ReadAllText(path));
            return manifest != null;
        }
        catch (Exception)
        {
            manifest = null;
            return false;
        }
    }
}
