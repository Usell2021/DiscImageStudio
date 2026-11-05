using System.IO;
using DiscImageStudio.Cd;
using DiscImageStudio.Core;

/// <summary>Uses only public CD APIs and no WPF, so it also runs outside the Windows test host.</summary>
internal static class CdCubicPitchTests
{
    private const double C1 = -0.01453, C2 = 0.02818, C3 = -0.02146;
    private const string OnePixelPng =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNgAAAAAgABc3UWCgAAAABJRU5ErkJggg==";

    internal static void Run()
    {
        TestZeroCoefficientsKeepConstantFormula();
        TestAgainstIndependentIntegration();
        TestScalarAndSimdAgree();
        TestValidation();
        TestSidecars();
        TestPresets();
        TestPhaseCorrection();
        Console.WriteLine("cd-cubic-pitch: zero identity, independent geometry, SIMD parity, validation, sidecars, presets and phase correction passed");
    }

    private static void TestZeroCoefficientsKeepConstantFormula()
    {
        CdDiscParameters disc = new(24.905, 57.807, 359848);
        double squaredRange = (57.807 * 57.807) - (24.905 * 24.905);
        foreach (long position in new[] { 0L, 1L, 123_456_789L, disc.TotalBytes - 1 })
        {
            double radius = Math.Sqrt((24.905 * 24.905) + ((double)position / disc.TotalBytes * squaredRange));
            double theta = 2.0 * 1200.0 * disc.TotalTimeSeconds / squaredRange * (radius - 24.905);
            (double actualRadius, double actualTheta) = disc.PolarFromByte(position);
            Require(actualRadius == radius && actualTheta == theta, "zero coefficients keep the constant formula bit for bit");
        }
    }

    private static void TestAgainstIndependentIntegration()
    {
        const double r0 = 24.897, r1 = 57.777;
        CdDiscParameters disc = new(r0, r1, 359848, PitchLinear: C1, PitchQuadratic: C2, PitchCubic: C3);
        double length = 359848 * 16.0;
        double Shape(double r)
        {
            double x = (r - r0) / (r1 - r0);
            return 1 + (C1 * x) + (C2 * x * x) + (C3 * x * x * x);
        }

        // Independent midpoint integration: thin-track arc and angle per unit radius.
        const int steps = 400_000;
        double step = (r1 - r0) / steps, unitArc = 0;
        for (int i = 0; i < steps; i++)
        {
            double r = r0 + ((i + .5) * step);
            unitArc += r / Shape(r) * step;
        }

        double growth = unitArc / length;
        double arc = 0, angle = 0;
        int check = 0;
        double[] checkAt = [.03, .25, .5, .75, .97];
        for (int i = 0; i < steps && check < checkAt.Length; i++)
        {
            double r = r0 + ((i + .5) * step);
            double nextArc = arc + (r / (growth * Shape(r)) * step);
            double nextAngle = angle + (step / (growth * Shape(r)));
            double target = checkAt[check] * length;
            if (nextArc >= target)
            {
                double t = (target - arc) / (nextArc - arc);
                (double radius, double theta) = disc.PolarFromByte((long)Math.Round(checkAt[check] * disc.TotalBytes));
                Require(Math.Abs(radius - (r0 + ((i + t) * step))) < 1e-5, "cubic radius matches independent integration");
                Require(Math.Abs(theta - (angle + (t * (nextAngle - angle)))) < 2e-3, "cubic angle matches independent integration");
                check++;
            }

            arc = nextArc;
            angle = nextAngle;
        }

        Require(check == checkAt.Length, "all independent checkpoints were reached");
        Require(disc.PolarFromByte(0) == (r0, 0), "cubic spiral starts at the inner radius");
        Require(Math.Abs(disc.RadiusFromByte(disc.TotalBytes) - r1) < 1e-9, "cubic spiral ends at the outer radius");
        (double middleRadius, double middleTheta) = disc.PolarFromByte(disc.TotalBytes / 2);
        Require(Math.Abs(disc.ThetaFromRadius(middleRadius) - middleTheta) < 1e-6, "angle by radius agrees with angle by byte");
        Require(disc with { InnerRadiusMm = 25 } != disc
            && (disc with { InnerRadiusMm = 25 }).RadiusFromByte(0) == 25, "with-copies never reuse stale geometry");
    }

    private static void TestScalarAndSimdAgree()
    {
        CdDiscParameters[] geometries =
        [
            new(24.5, 57.5, 64, LinearVelocityMmPerSecond: 6_000_000, PitchLinear: C1, PitchQuadratic: C2, PitchCubic: C3),
            new(24.5, 57.5, 64, LinearVelocityMmPerSecond: 6_000_000, StartAngleRadians: -1, PitchLinear: .2),
            new(24.5, 57.5, 64, LinearVelocityMmPerSecond: 6_000_000, PhaseCorrectionDegrees: [3, -2, 5]),
            new(24.5, 57.5, 64, LinearVelocityMmPerSecond: 6_000_000, PitchLinear: C1, PhaseCorrectionDegrees: [3, -2, 5, 1]),
        ];
        foreach (CdDiscParameters disc in geometries)
        foreach (bool interleave in new[] { false, true })
        {
            static byte Pattern(double x, double y)
                => ((BitConverter.DoubleToInt64Bits(x) ^ BitConverter.DoubleToInt64Bits(y)) & 1) == 0 ? (byte)0 : (byte)255;
            using MemoryStream scalar = new(), simd = new();
            CdTrackGenerator.GeneratePatternToStream(Pattern, scalar, disc, interleave, useSimdSampling: false);
            CdTrackGenerator.GeneratePatternToStream(Pattern, simd, disc, interleave, useSimdSampling: true);
            Require(scalar.ToArray().AsSpan().SequenceEqual(simd.ToArray()), "cubic SIMD matches scalar coordinates exactly");
        }
    }

    private static void TestValidation()
    {
        foreach (CdDiscParameters invalid in new CdDiscParameters[]
        {
            new(24, 58, 1000, PitchLinear: -2),
            new(24, 58, 1000, PitchCubic: double.NaN),
        })
        {
            bool rejected = false;
            try { invalid.Validate(); }
            catch (ArgumentException) { rejected = true; }
            Require(rejected, "a non-positive or non-finite pitch shape is rejected");
        }
    }

    private static void TestSidecars()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"disc-cd-cubic-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string image = Path.Combine(directory, "pixel.png");
            File.WriteAllBytes(image, Convert.FromBase64String(OnePixelPng));
            string constant = Path.Combine(directory, "constant.wav");
            CdTrackGenerator.Generate(image, constant, new CdDiscParameters(24, 58, 8), false);
            Require(!File.ReadAllText(constant + ".json").Contains("pitch", StringComparison.OrdinalIgnoreCase),
                "constant-pitch sidecars stay unchanged");
            string cubic = Path.Combine(directory, "cubic.wav");
            CdTrackGenerator.Generate(image, cubic,
                new CdDiscParameters(24, 58, 8, PitchLinear: C1, PitchQuadratic: C2, PitchCubic: C3), false);
            CdTrackMetadata metadata = CdTrackMetadata.TryLoad(cubic)
                ?? throw new InvalidOperationException("Cubic sidecar did not load.");
            Require(metadata.PitchLinear == C1 && metadata.PitchQuadratic == C2 && metadata.PitchCubic == C3,
                "cubic sidecar round-trips its coefficients");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void TestPresets()
    {
        const string id = "noname-cd-rw-infodisc";
        DiscPresetJsonDocument builtIn = DiscPresetJsonStore.LoadBuiltIn();
        CdDiscPresetDefinition preset = builtIn.CdPresets.Single(value => value.Id == id);
        Require(preset.PitchLinear != 0 && preset.PitchQuadratic != 0 && preset.PitchCubic != 0,
            "the measured preset carries its pitch shape");
        Require(builtIn.CdPresets.Where(value => value.Id != id)
                .All(value => value is { PitchLinear: 0, PitchQuadratic: 0, PitchCubic: 0 }),
            "the other CD presets stay constant pitch");

        string directory = Path.Combine(Path.GetTempPath(), $"disc-cd-preset-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string saved = Path.Combine(directory, "presets.json");
            DiscPresetJsonDocument document = builtIn with
            {
                CdPresets = builtIn.CdPresets.Where(value => value.Id is id or "cd-80").ToList(),
                DvdPresets = [],
                BluRayPresets = [],
            };
            DiscPresetJsonStore.Save(saved, document);
            CdDiscPresetDefinition reloaded = DiscPresetJsonStore.Load(saved).CdPresets.Single(value => value.Id == id);
            Require(reloaded.PitchLinear == preset.PitchLinear && reloaded.PitchQuadratic == preset.PitchQuadratic
                && reloaded.PitchCubic == preset.PitchCubic, "the pitch shape survives a preset file round trip");
            using (System.Text.Json.JsonDocument json = System.Text.Json.JsonDocument.Parse(File.ReadAllText(saved)))
            {
                foreach (System.Text.Json.JsonElement value in json.RootElement.GetProperty("cdPresets").EnumerateArray())
                {
                    Require(value.GetProperty("id").GetString() == id || !value.TryGetProperty("pitchLinear", out _),
                        "constant-pitch presets keep their existing JSON shape");
                }
            }

            bool rejected = false;
            try
            {
                DiscPresetJsonStore.Validate(document with
                {
                    CdPresets = [preset with { PitchLinear = -2 }],
                });
            }
            catch (InvalidDataException) { rejected = true; }
            Require(rejected, "a CD preset with a non-positive pitch shape is rejected");

            string image = Path.Combine(directory, "pixel.png");
            File.WriteAllBytes(image, Convert.FromBase64String(OnePixelPng));
            CdDiscModule module = new();
            string track = Path.Combine(directory, "track.wav");
            // Only the sector count and outer radius are overridden, which keeps the track small
            // and shows that explicit options win while the preset supplies everything else.
            module.ExecuteAsync(new DiscJobRequest("cd-generate",
            [
                "--input", image, "--output", track, "--preset", id, "--sectors", "8", "--r1", "58",
            ])).GetAwaiter().GetResult();
            CdTrackMetadata metadata = CdTrackMetadata.TryLoad(track)
                ?? throw new InvalidOperationException("Preset track sidecar did not load.");
            Require(metadata.Sectors == 8 && metadata.OuterRadiusMm == 58, "explicit options override the preset");
            Require(metadata.InnerRadiusMm == preset.InnerRadiusMm && metadata.PitchLinear == preset.PitchLinear
                && metadata.PitchQuadratic == preset.PitchQuadratic && metadata.PitchCubic == preset.PitchCubic,
                "--preset supplies the geometry the options left out");

            bool unknown = false;
            try
            {
                module.ExecuteAsync(new DiscJobRequest("cd-generate",
                [
                    "--input", image, "--output", track, "--preset", "no-such-preset",
                ])).GetAwaiter().GetResult();
            }
            catch (ArgumentException exception) when (exception.Message.Contains(id, StringComparison.Ordinal))
            {
                unknown = true;
            }
            Require(unknown, "an unknown preset is rejected and lists the available ids");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void TestPhaseCorrection()
    {
        double[] table = [10, -20, 40];
        foreach (CdDiscParameters plain in new CdDiscParameters[]
        {
            new(24.9, 57.8, 359848),
            new(24.9, 57.8, 359848, PitchLinear: C1, PitchQuadratic: C2, PitchCubic: C3),
        })
        {
            CdDiscParameters corrected = plain with { PhaseCorrectionDegrees = table };
            corrected.Validate();
            foreach ((double fraction, double degrees) in new[] { (0.0, 10.0), (.25, -5.0), (.5, -20.0), (.75, 10.0), (1.0, 40.0) })
            {
                long position = (long)(fraction * plain.TotalBytes);
                (double radius, double theta) = plain.PolarFromByte(position);
                (double correctedRadius, double correctedTheta) = corrected.PolarFromByte(position);
                Require(correctedRadius == radius, "phase correction leaves radii alone");
                Require(Math.Abs(correctedTheta - theta - (degrees * Math.PI / 180)) < 1e-9,
                    "phase correction interpolates evenly over the track");
            }
        }

        foreach (double[] invalid in new[] { new[] { 1.0 }, new[] { 1.0, double.NaN } })
        {
            bool rejected = false;
            try { new CdDiscParameters(24, 58, 1000, PhaseCorrectionDegrees: invalid).Validate(); }
            catch (ArgumentException) { rejected = true; }
            Require(rejected, "an unusable phase correction is rejected");
        }

        string directory = Path.Combine(Path.GetTempPath(), $"disc-cd-phase-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string image = Path.Combine(directory, "pixel.png");
            File.WriteAllBytes(image, Convert.FromBase64String(OnePixelPng));
            string track = Path.Combine(directory, "track.wav");
            CdTrackGenerator.Generate(image, track, new CdDiscParameters(24, 58, 8, PhaseCorrectionDegrees: table), false);
            CdTrackMetadata metadata = CdTrackMetadata.TryLoad(track)
                ?? throw new InvalidOperationException("Phase sidecar did not load.");
            Require(metadata.PhaseCorrectionDegrees is { } saved && saved.SequenceEqual(table),
                "the sidecar round-trips the phase correction");

            const string id = "noname-cd-rw-infodisc";
            DiscPresetJsonDocument builtIn = DiscPresetJsonStore.LoadBuiltIn();
            CdDiscPresetDefinition preset = builtIn.CdPresets.Single(value => value.Id == id);
            Require(preset.PhaseCorrectionDegrees is { Length: > 2 }, "the measured preset carries its phase table");
            string viaPreset = Path.Combine(directory, "preset.wav");
            new CdDiscModule().ExecuteAsync(new DiscJobRequest("cd-generate",
            [
                "--input", image, "--output", viaPreset, "--preset", id, "--sectors", "8",
            ])).GetAwaiter().GetResult();
            Require(CdTrackMetadata.TryLoad(viaPreset)?.PhaseCorrectionDegrees is { } applied
                && applied.SequenceEqual(preset.PhaseCorrectionDegrees!), "--preset applies the phase table");

            bool rejected = false;
            try
            {
                DiscPresetJsonStore.Validate(builtIn with
                {
                    CdPresets = [preset with { PhaseCorrectionDegrees = [double.PositiveInfinity, 0] }],
                });
            }
            catch (InvalidDataException) { rejected = true; }
            Require(rejected, "a preset with an unusable phase table is rejected");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}
