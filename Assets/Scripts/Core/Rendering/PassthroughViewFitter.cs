using Meta.XR;
using UnityEngine;

/// <summary>
/// Sizes and positions one eye's passthrough quad so the camera image appears at its
/// true 1:1 angular size and direction, derived from the live intrinsics (including
/// the sensor crop of the active resolution and the lens tilt) instead of hand-tuned
/// transforms. Correct scale and stereo disparity take priority over covering the
/// full eye view: the display FOV is larger than the camera FOV, so a border remains.
/// Refits automatically when the camera resolution changes.
/// </summary>
public class PassthroughViewFitter : MonoBehaviour
{
    [SerializeField] private PassthroughCameraAccess cameraAccess;

    [Tooltip("Eye anchor this quad belongs to (LeftEyeAnchor/RightEyeAnchor). The quad is placed relative to it.")]
    [SerializeField] private Transform eyeAnchor;

    [Tooltip("Distance of the quad plane in meters. Effectively infinity (zero surface " +
             "parallax; perceived depth comes from the stereo image content). Must stay " +
             "in front of the camera far clip plane (1000).")]
    [SerializeField] private float quadDistance = 900f;

    [Tooltip("Manual fine-tune of the image size. 1 = true 1:1 angular scale; values above 1 " +
             "magnify and progressively distort stereo depth and size perception.")]
    [SerializeField] private float imageScale = 1f;

    private Vector2Int fittedResolution;

    private void LateUpdate()
    {
        if (cameraAccess == null || !cameraAccess.IsPlaying || eyeAnchor == null)
        {
            return;
        }
        if (cameraAccess.CurrentResolution == fittedResolution)
        {
            return;
        }
        if (Fit())
        {
            fittedResolution = cameraAccess.CurrentResolution;
        }
    }

    private bool Fit()
    {
        var intrinsics = cameraAccess.Intrinsics;
        Vector2 sensor = intrinsics.SensorResolution;
        Vector2 focal = intrinsics.FocalLength;
        Vector2 principal = intrinsics.PrincipalPoint;
        if (sensor.x <= 0f || sensor.y <= 0f || focal.x <= 0f || focal.y <= 0f)
        {
            return false;
        }

        // The active resolution is a centered crop of the sensor (mirrors the SDK's
        // internal CalcSensorCropRegion): 1280x960 crops the square sensor vertically.
        Vector2 current = cameraAccess.CurrentResolution;
        var scale = new Vector2(current.x / sensor.x, current.y / sensor.y);
        scale /= Mathf.Max(scale.x, scale.y);
        var cropMin = Vector2.Scale(sensor, (Vector2.one - scale) * 0.5f);
        var cropMax = cropMin + Vector2.Scale(sensor, scale);

        // Signed view-space tangents of the camera image edges around the optical axis.
        float camLeft = (cropMin.x - principal.x) / focal.x;
        float camRight = (cropMax.x - principal.x) / focal.x;
        float camBottom = (cropMin.y - principal.y) / focal.y;
        float camTop = (cropMax.y - principal.y) / focal.y;

        // Orient along the camera's optical axis (LensOffset.rotation is what the SDK's
        // GetCameraPose composes onto the head pose) and offset the quad center by the
        // principal-point asymmetry, so the image sits in its true direction. The lens
        // position offset (~cm) is irrelevant at quadDistance and deliberately ignored.
        var lensRotation = intrinsics.LensOffset.rotation;
        var centerInAnchor = lensRotation * new Vector3(
            (camRight + camLeft) * 0.5f,
            (camTop + camBottom) * 0.5f,
            1f) * quadDistance;
        transform.SetPositionAndRotation(
            eyeAnchor.TransformPoint(centerInAnchor),
            eyeAnchor.rotation * lensRotation);
        // Parent chain under the eye anchor has unit scale, so localScale is world-sized.
        transform.localScale = new Vector3(
            (camRight - camLeft) * imageScale * quadDistance,
            (camTop - camBottom) * imageScale * quadDistance,
            1f);
        return true;
    }
}
