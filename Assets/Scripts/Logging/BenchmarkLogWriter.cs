using System;
using System.IO;
using System.Threading.Tasks;

/// <summary>
/// Writes BenchmarkSession data to persistent storage as individual JSON files.
/// Files are stored under the benchmark_logs folder with naming:
///   benchmark_{sessionId}.json  (e.g. benchmark_shader_20260308_112500.json)
/// </summary>
public static class BenchmarkLogWriter
{
    /// <summary>
    /// Writes a benchmark session to a JSON file in the specified folder asynchronously.
    /// </summary>
    public static async void WriteLog(string folderPath, BenchmarkSession session, Logger logger = null)
    {
        try
        {
            Directory.CreateDirectory(folderPath);

            string fileId = session.SessionId ?? session.RequestId;
            string fileName = $"benchmark_{fileId}.json";
            string filePath = Path.Combine(folderPath, fileName);

            // Using Task.Run to offload JSON serialization and file IO
            await Task.Run(() => 
            {
                string json = session.ToJson();
                File.WriteAllText(filePath, json);
            });

            // This runs on the original synchronization context (main thread in Unity)
            logger?.LogDebug($"Benchmark written to {filePath}", nameof(BenchmarkLogWriter));
        }
        catch (Exception ex)
        {
            logger?.LogError($"Failed to write benchmark log: {ex.Message}", nameof(BenchmarkLogWriter));
        }
    }
}
