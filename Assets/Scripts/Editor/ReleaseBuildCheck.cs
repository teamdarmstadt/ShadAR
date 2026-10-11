using System.IO;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

/// <summary>
/// Stops a non-development build that would ship secrets. Everything under Assets/Resources is
/// packed into the APK whether git tracks it or not, and API keys belong in the device config
/// files, never in the build.
/// </summary>
public class ReleaseBuildCheck : IPreprocessBuildWithReport
{
    private const string DevAgentSettings = "Assets/Resources/DevAgentSettings.asset";
    private const string Scene = "Assets/Scenes/PromptedVision.unity";

    public int callbackOrder => 0;

    public void OnPreprocessBuild(BuildReport report)
    {
        if ((report.summary.options & BuildOptions.Development) != 0) return;

        if (File.Exists(DevAgentSettings))
        {
            Fail($"{DevAgentSettings} holds this machine's dev-agent address and access token. " +
                 "Delete it before a release build (Unity recreates it when the dev agent is used).");
        }

        var openAI = Resources.Load<OpenAI.OpenAIConfiguration>("OpenAIConfiguration");
        if (openAI != null && !string.IsNullOrWhiteSpace(openAI.ApiKey))
        {
            Fail("Assets/Resources/OpenAIConfiguration.asset contains an API key. Clear it; " +
                 "keys go into openai_config.json on the headset.");
        }

        if (File.Exists(Scene) && Regex.IsMatch(File.ReadAllText(Scene), @"defaultApiKey: *\S"))
        {
            Fail("ClaudeConfigManager.defaultApiKey in the scene is set. Clear it; keys go into " +
                 "claude_config.json on the headset.");
        }
    }

    private static void Fail(string reason)
    {
        throw new BuildFailedException($"Release build stopped: {reason}");
    }
}
