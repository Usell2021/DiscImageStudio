using System.Collections.Concurrent;
using System.Text.Json.Serialization;
using DiscImageStudio.Core;

namespace DiscImageStudio.Cd;

/// <summary>
/// CD spiral. Zero pitch coefficients keep the constant-pitch formulas; any other value uses
/// dr/dθ = b (1 + c1 x + c2 x² + c3 x³), x = (r - r0) / (r1 - r0), through
/// <see cref="DvdTrackGeometry"/>. Zero coefficients are left out of sidecars.
/// <see cref="PhaseCorrectionDegrees"/> adds a measured angle on top of either spiral: its
/// values are evenly spaced over the track, from the first byte to the end, and interpolated
/// linearly. It describes groove wander that no smooth pitch can follow; radii are unaffected.
/// </summary>
public sealed record CdDiscParameters(
    double InnerRadiusMm,
    double OuterRadiusMm,
    long Sectors,
    double LinearVelocityMmPerSecond = 1200.0,
    double StartAngleRadians = 0.0,
    double ImageOuterRadiusMm = 57.5,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)] double PitchLinear = 0,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)] double PitchQuadratic = 0,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)] double PitchCubic = 0,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] double[]? PhaseCorrectionDegrees = null)
{
    // Records copy and compare every instance field, so an instance cache would go stale
    // across `with` and break value equality. Parameter sets per process are few.
    private static readonly ConcurrentDictionary<
        (double, double, double, double, double, double), DvdTrackGeometry> Geometries = new();

    public const int BytesPerSector = 2352;

    /// <summary>ECMA-130 CLV scanning speed; the UI no longer exposes it as a parameter.</summary>
    public const double StandardLinearVelocityMmPerSecond = 1200.0;

    /// <summary>Radius on the source image canvas that maps to the disc outer edge.</summary>
    public const double StandardImageOuterRadiusMm = 57.5;

    public long TotalBytes => checked(Sectors * BytesPerSector);

    public double TotalTimeSeconds => Sectors / 75.0;

    internal bool IsConstantPitch => PitchLinear == 0 && PitchQuadratic == 0 && PitchCubic == 0;

    internal double TrackLengthMm => LinearVelocityMmPerSecond * TotalTimeSeconds;

    internal DvdTrackGeometry VariablePitchGeometry => Geometries.GetOrAdd(
        (InnerRadiusMm, OuterRadiusMm, TrackLengthMm, PitchLinear, PitchQuadratic, PitchCubic),
        static key => DvdTrackGeometry.Create(key.Item1, key.Item2, key.Item3, key.Item4, key.Item5, key.Item6));

    /// <summary>Added last so a track without a table keeps its angles bit for bit.</summary>
    internal double WithPhaseCorrection(double angleRadians, double fraction)
    {
        if (PhaseCorrectionDegrees is not { } table)
        {
            return angleRadians;
        }

        double position = fraction * (table.Length - 1);
        int index = Math.Min((int)position, table.Length - 2);
        double degrees = table[index] + ((table[index + 1] - table[index]) * (position - index));
        return angleRadians + (degrees * Math.PI / 180.0);
    }

    private double RadiusSquaredRange =>
        (OuterRadiusMm * OuterRadiusMm) - (InnerRadiusMm * InnerRadiusMm);

    public void Validate()
    {
        if (Sectors <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(Sectors), "CD sector count must be positive.");
        }

        if (!double.IsFinite(InnerRadiusMm)
            || !double.IsFinite(OuterRadiusMm)
            || InnerRadiusMm <= 0
            || OuterRadiusMm <= InnerRadiusMm)
        {
            throw new ArgumentOutOfRangeException(
                nameof(OuterRadiusMm),
                "CD outer radius must be greater than its positive inner radius.");
        }

        if (!double.IsFinite(LinearVelocityMmPerSecond) || LinearVelocityMmPerSecond <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(LinearVelocityMmPerSecond),
                "CD linear velocity must be positive.");
        }

        if (!double.IsFinite(StartAngleRadians))
        {
            throw new ArgumentOutOfRangeException(nameof(StartAngleRadians));
        }

        if (!double.IsFinite(ImageOuterRadiusMm) || ImageOuterRadiusMm <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(ImageOuterRadiusMm),
                "Image outer radius must be positive.");
        }

        _ = TotalBytes;

        if (!IsConstantPitch)
        {
            DvdTrackGeometry.ValidatePitchCoefficients(PitchLinear, PitchQuadratic, PitchCubic);
            _ = VariablePitchGeometry;
        }

        if (PhaseCorrectionDegrees is { } table
            && (table.Length < 2 || !table.All(double.IsFinite)))
        {
            throw new ArgumentOutOfRangeException(
                nameof(PhaseCorrectionDegrees),
                "A CD phase correction needs at least two finite values.");
        }
    }

    public double RadiusFromByte(long globalByte)
    {
        double fraction = Math.Clamp((double)globalByte / TotalBytes, 0.0, 1.0);
        if (!IsConstantPitch)
        {
            return VariablePitchGeometry.AtArcLengthFast(fraction * TrackLengthMm).Radius;
        }

        return Math.Sqrt(
            (InnerRadiusMm * InnerRadiusMm) + (fraction * RadiusSquaredRange));
    }

    public double ThetaFromRadius(double radiusMm)
        => !IsConstantPitch
            ? StartAngleRadians + VariablePitchGeometry.AngleAtRadius(radiusMm)
            : StartAngleRadians
            + ((2.0 * LinearVelocityMmPerSecond * TotalTimeSeconds / RadiusSquaredRange)
                * (radiusMm - InnerRadiusMm));

    public (double RadiusMm, double ThetaRadians) PolarFromByte(long globalByte)
    {
        if (!IsConstantPitch)
        {
            double fraction = Math.Clamp((double)globalByte / TotalBytes, 0.0, 1.0);
            (double variableRadius, double angle) = VariablePitchGeometry.AtArcLengthFast(fraction * TrackLengthMm);
            return (variableRadius, WithPhaseCorrection(StartAngleRadians + angle, fraction));
        }

        double radius = RadiusFromByte(globalByte);
        return (radius, WithPhaseCorrection(
            ThetaFromRadius(radius),
            Math.Clamp((double)globalByte / TotalBytes, 0.0, 1.0)));
    }

    public (double X, double Y) ImagePointFromByte(long globalByte, int imageSize)
    {
        (double radius, double theta) = PolarFromByte(globalByte);
        double imageRadius = imageSize / 2.0;
        double pixelRadius = imageRadius * radius / ImageOuterRadiusMm;
        double centre = imageSize / 2.0;
        return (
            centre + (pixelRadius * Math.Cos(theta)),
            centre + (pixelRadius * Math.Sin(theta)));
    }
}
