using Meta.XR;
using UnityEngine;

/// <summary>
/// Reprojects a normalized viewport point from one passthrough camera into the other.
/// Reprojection into the SAME camera is depth-independent, so only the cross-eye direction
/// needs a depth estimate — and its accuracy directly determines how well overlays line up
/// in the second eye. Depth acquisition (Depth-API raycast) and reprojection are separate so
/// callers can cache/smooth depth instead of raycasting every frame.
/// Used by DetectionTracker to shift left-camera detection boxes into the right eye.
/// </summary>
public static class StereoRectReprojector
{
    /// <summary>
    /// Real depth (m) of the environment along the ray through <paramref name="viewportPoint"/>
    /// of <paramref name="fromCam"/>, via Depth-API raycast. False when the raycaster is
    /// unavailable (unsupported, disabled, no permission) or the ray misses.
    /// </summary>
    public static bool TryRaycastDepth(Vector2 viewportPoint, PassthroughCameraAccess fromCam,
        EnvironmentRaycastManager raycaster, out float depthMeters)
    {
        depthMeters = 0f;
        if (fromCam == null || !fromCam.IsPlaying || raycaster == null || !raycaster.isActiveAndEnabled)
        {
            return false;
        }

        Ray ray = fromCam.ViewportPointToRay(viewportPoint);
        if (!raycaster.Raycast(ray, out EnvironmentRaycastHit hit, 20f))
        {
            return false;
        }

        depthMeters = Vector3.Distance(ray.origin, hit.point);
        return true;
    }

    /// <summary>
    /// Maps a point (normalized viewport of <paramref name="fromCam"/>, origin bottom-left) into
    /// <paramref name="toCam"/>'s viewport, assuming the content there sits at
    /// <paramref name="depthMeters"/>. Used on box centers; the box size is kept, as the per-eye
    /// size difference across a 6.4 cm baseline is negligible.
    /// </summary>
    public static Vector2 Reproject(Vector2 viewportPoint, PassthroughCameraAccess fromCam, PassthroughCameraAccess toCam,
        float depthMeters)
    {
        if (fromCam == null || toCam == null || !fromCam.IsPlaying || !toCam.IsPlaying)
        {
            return viewportPoint;
        }

        Ray ray = fromCam.ViewportPointToRay(viewportPoint);
        return toCam.WorldToViewportPoint(ray.GetPoint(Mathf.Max(depthMeters, 0.1f)));
    }
}
