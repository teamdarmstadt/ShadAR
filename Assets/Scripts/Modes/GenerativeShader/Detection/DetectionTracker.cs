using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

/// <summary>
/// Holds the latest detector result until the next one arrives or it goes stale, and hands it
/// out per eye. Detections are 2D boxes in LEFT-camera viewport space (inference runs on the left
/// feed), so the left eye uses them as they are; the right eye gets each box shifted by its
/// disparity through <see cref="StereoRectReprojector"/>, using real depth from the Depth API.
/// </summary>
public class DetectionTracker : MonoBehaviour
{
    // How long a detection's sampled depth stays valid before it is raycast again.
    private const float StereoDepthMaxAgeSeconds = 0.25f;

    [SerializeField, Tooltip("Seconds without a new detector result before the boxes are dropped")]
    private float maxDetectionAge = 1f;

    [Header("Camera References")]
    [SerializeField] private PassthroughCameraAccess leftCameraAccess;
    [SerializeField] private PassthroughCameraAccess rightCameraAccess;

    [Header("Stereo Reprojection")]
    [Tooltip("Depth-API raycaster that finds the real depth of each detection so its box can be " +
             "shifted correctly for the right eye. Optional — without it the fallback depth is used.")]
    [SerializeField] private EnvironmentRaycastManager raycastManager;
    [Tooltip("Assumed detection depth (m) when the environment raycast is unavailable or misses.")]
    [SerializeField, Range(0.3f, 10f)] private float stereoFallbackDepthMeters = 1.5f;

    private readonly List<DetectedObject> currentDetections = new();
    private readonly List<DetectedObject> rightEyeDetections = new();
    private readonly List<float> stereoDepths = new();
    private float lastUpdateTime;
    private float stereoDepthTime;
    private int rightEyeFrame = -1;

    /// <summary>The latest detections in left-camera viewport space.</summary>
    public List<DetectedObject> CurrentFrameDetections => currentDetections;

    public void UpdateFromInference(List<DetectedObject> detections)
    {
        currentDetections.Clear();
        if (detections != null) currentDetections.AddRange(detections);

        lastUpdateTime = Time.time;
        stereoDepths.Clear(); // new boxes, new depths
        rightEyeFrame = -1;
    }

    private void Update()
    {
        if (currentDetections.Count > 0 && Time.time - lastUpdateTime > maxDetectionAge)
        {
            currentDetections.Clear();
            stereoDepths.Clear();
        }
    }

    /// <summary>
    /// Detections in the viewport space of the given eye's camera. The returned list is reused;
    /// read it, do not keep it.
    /// </summary>
    public List<DetectedObject> GetDetectionsForEye(PassthroughCameraAccess eyeCameraAccess)
    {
        if (eyeCameraAccess != rightCameraAccess || rightCameraAccess == leftCameraAccess)
        {
            return currentDetections;
        }

        if (rightEyeFrame == Time.frameCount) return rightEyeDetections;
        rightEyeFrame = Time.frameCount;

        // ponytail: depth is sampled per detection at most every StereoDepthMaxAgeSeconds — one
        // raycast per detection per frame was measurable CPU, and object depth changes slowly.
        bool refreshDepth = stereoDepths.Count != currentDetections.Count
                            || Time.unscaledTime - stereoDepthTime > StereoDepthMaxAgeSeconds;
        if (refreshDepth)
        {
            stereoDepths.Clear();
            stereoDepthTime = Time.unscaledTime;
        }

        rightEyeDetections.Clear();
        for (int i = 0; i < currentDetections.Count; i++)
        {
            var detection = currentDetections[i];
            var box = detection.BoundingBox;
            // BoundingBox.position is the box CENTER (see DetectedObject), not Rect's min corner.
            if (refreshDepth)
            {
                stereoDepths.Add(StereoRectReprojector.TryRaycastDepth(
                    box.position, leftCameraAccess, raycastManager, out float depth)
                    ? depth
                    : stereoFallbackDepthMeters);
            }

            Vector2 center = StereoRectReprojector.Reproject(box.position, leftCameraAccess, rightCameraAccess, stereoDepths[i]);
            detection.BoundingBox = new Rect(center, box.size);
            rightEyeDetections.Add(detection);
        }

        return rightEyeDetections;
    }
}
