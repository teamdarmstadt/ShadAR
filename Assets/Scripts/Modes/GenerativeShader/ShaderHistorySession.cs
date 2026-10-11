using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// One shader generation: a folder under persistentDataPath/shader_history holding the
/// transcription, the generated shader source, its detection manifest and the compiled bundle.
/// </summary>
public class ShaderHistorySession
{
    private const string HistoryBaseFolderName = "shader_history";
    private const string TranscriptionFileName = "transcription.txt";
    private const string ShaderFileName = "GeneratedShader.shader";
    private const string BundleFileName = "shader.bundle";
    private const string SessionIdFormat = "yyyyMMdd_HHmmss";
    private const string SessionFolderPrefix = "shader_";

    /// <summary>Folder name of this session, e.g. "shader_20260308_112500".</summary>
    public string FolderName { get; }

    public string SessionFolderPath { get; }
    public string TranscriptionPath { get; }
    public string ShaderFilePath { get; }
    public string BundlePath { get; }

    private ShaderHistorySession(string folderName)
    {
        FolderName = folderName;
        SessionFolderPath = Path.Combine(GetHistoryBaseFolderPath(), folderName);
        Directory.CreateDirectory(SessionFolderPath);

        TranscriptionPath = Path.Combine(SessionFolderPath, TranscriptionFileName);
        ShaderFilePath = Path.Combine(SessionFolderPath, ShaderFileName);
        BundlePath = Path.Combine(SessionFolderPath, BundleFileName);
    }

    /// <summary>
    /// Starts a new session with a timestamp-based folder name.
    /// </summary>
    public static ShaderHistorySession StartNewSession()
    {
        return new ShaderHistorySession(SessionFolderPrefix + DateTime.Now.ToString(SessionIdFormat));
    }

    public void SaveTranscription(string transcriptionText)
    {
        File.WriteAllText(TranscriptionPath, transcriptionText);
    }

    public void SaveShaderCode(string shaderCode)
    {
        File.WriteAllText(ShaderFilePath, shaderCode);
    }

    /// <summary>
    /// Saves the open-vocabulary detection manifest next to the shader so it can be resolved
    /// later from the shader history.
    /// </summary>
    public void SaveDetectionClasses(ShaderDetectionManifest manifest)
    {
        manifest?.Save(SessionFolderPath);
    }

    /// <summary>
    /// Saves the compiled bundle without blocking the main thread.
    /// </summary>
    public async Task SaveBundleAsync(byte[] bundleData)
    {
        using var fs = new FileStream(BundlePath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, useAsync: true);
        await fs.WriteAsync(bundleData, 0, bundleData.Length);
    }

    public static string GetHistoryBaseFolderPath()
    {
        return Path.Combine(Application.persistentDataPath, HistoryBaseFolderName);
    }
}
