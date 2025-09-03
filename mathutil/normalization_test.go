package mathutil

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// ============================================================================
// TEST HELPERS
// ============================================================================

func generateRandomData(samples, features int) [][]float64 {
	data := make([][]float64, samples)
	for i := range data {
		data[i] = make([]float64, features)
		for j := range data[i] {
			data[i][j] = rand.Float64()*100 - 50 // Random values between -50 and 50
		}
	}
	return data
}

func assertFloatSlicesEqual(t *testing.T, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("slice length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tolerance {
			t.Errorf("mismatch at index %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

// =============================================================================
// UNIT TESTS
// =============================================================================

func TestNewNormalizer(t *testing.T) {
	t.Run("initialization", func(t *testing.T) {
		t.Parallel()
		norm := NewNormalizer()

		if norm == nil {
			t.Fatal("NewNormalizer() returned nil")
		}

		if norm.fitted {
			t.Error("new normalizer should not be fitted")
		}

		if norm.mean != nil {
			t.Error("new normalizer should have nil mean")
		}

		if norm.stddev != nil {
			t.Error("new normalizer should have nil stddev")
		}
	})
}

func TestNormalizerFit(t *testing.T) {
	t.Run("valid_data", func(t *testing.T) {
		tests := []struct {
			name           string
			data           [][]float64
			expectedMean   []float64
			expectedStddev []float64
		}{
			{
				name: "simple_3x2_data",
				data: [][]float64{
					{1.0, 10.0},
					{2.0, 20.0},
					{3.0, 30.0},
				},
				expectedMean:   []float64{2.0, 20.0},
				expectedStddev: []float64{math.Sqrt(2.0 / 3.0), math.Sqrt(200.0 / 3.0)},
			},
			{
				name: "single_sample",
				data: [][]float64{
					{5.0, 15.0, 25.0},
				},
				expectedMean:   []float64{5.0, 15.0, 25.0},
				expectedStddev: []float64{1.0, 1.0, 1.0}, // Should be 1.0 for constant features
			},
			{
				name: "zero_data",
				data: [][]float64{
					{0.0, 0.0},
					{0.0, 0.0},
				},
				expectedMean:   []float64{0.0, 0.0},
				expectedStddev: []float64{1.0, 1.0}, // Should be 1.0 to avoid division by zero
			},
			{
				name: "negative_values",
				data: [][]float64{
					{-1.0, -10.0},
					{-2.0, -20.0},
					{-3.0, -30.0},
				},
				expectedMean:   []float64{-2.0, -20.0},
				expectedStddev: []float64{math.Sqrt(2.0 / 3.0), math.Sqrt(200.0 / 3.0)},
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				norm := NewNormalizer()

				err := norm.Fit(tc.data)
				if err != nil {
					t.Fatalf("Fit failed: %v", err)
				}

				if !norm.fitted {
					t.Error("normalizer should be marked as fitted")
				}

				assertFloatSlicesEqual(t, norm.mean, tc.expectedMean, 1e-10)
				assertFloatSlicesEqual(t, norm.stddev, tc.expectedStddev, 1e-10)
			})
		}
	})

	t.Run("invalid_data", func(t *testing.T) {
		tests := []struct {
			name string
			data [][]float64
		}{
			{"empty_data", [][]float64{}},
			{"empty_sample", [][]float64{{}}},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				norm := NewNormalizer()
				err := norm.Fit(tc.data)

				if err == nil {
					t.Errorf("expected error for %s, but got nil", tc.name)
				}

				if norm.fitted {
					t.Error("normalizer should not be fitted after error")
				}
			})
		}
	})

	t.Run("special_values", func(t *testing.T) {
		t.Parallel()
		tests := []struct {
			name           string
			data           [][]float64
			expectedMean   []float64
			expectedStddev []float64
		}{
			{
				name: "with_nan",
				data: [][]float64{
					{1.0, math.NaN()},
					{2.0, 3.0},
				},
				expectedMean:   []float64{1.5, math.NaN()}, // Normal calc, NaN propagates
				expectedStddev: []float64{0.5, math.NaN()}, // Normal calc, NaN propagates
			},
			{
				name: "with_inf_different_sign",
				data: [][]float64{
					{1.0, math.Inf(1)},
					{2.0, math.Inf(-1)},
				},
				expectedMean:   []float64{1.5, math.NaN()}, // Normal, (+Inf + -Inf)/2 = NaN
				expectedStddev: []float64{0.5, math.NaN()}, // Normal, NaN from mean propagates
			},
			{
				name: "with_inf_same_sign_positive",
				data: [][]float64{
					{1.0, math.Inf(1)},
					{2.0, math.Inf(1)},
				},
				expectedMean:   []float64{1.5, math.Inf(1)}, // Normal, (+Inf+Inf)/2=+Inf
				expectedStddev: []float64{0.5, math.NaN()},  // Normal, (+Inf - +Inf)² = NaN
			},
			{
				name: "with_inf_same_sign_negative",
				data: [][]float64{
					{1.0, math.Inf(-1)},
					{2.0, math.Inf(-1)},
				},
				expectedMean:   []float64{1.5, math.Inf(-1)}, // Normal, (-Inf + -Inf)/2 = -Inf
				expectedStddev: []float64{0.5, math.NaN()},   // Normal, (-Inf - -Inf)² = NaN
			},
			{
				name: "mixed_inf_and_finite",
				data: [][]float64{
					{1.0, math.Inf(1)},
					{2.0, 5.0},
					{3.0, 10.0},
				},
				expectedMean:   []float64{2.0, math.Inf(1)},                 // Normal, (+Inf + 5 + 10)/3 = +Inf
				expectedStddev: []float64{math.Sqrt(2.0 / 3.0), math.NaN()}, // Normal, any diff with +Inf gives NaN
			},
			{
				name: "multiple_nans",
				data: [][]float64{
					{math.NaN(), 1.0},
					{math.NaN(), 2.0},
				},
				expectedMean:   []float64{math.NaN(), 1.5}, // NaN propagates, normal calc
				expectedStddev: []float64{math.NaN(), 0.5}, // NaN propagates, normal calc
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				norm := NewNormalizer()
				err := norm.Fit(tc.data)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}

				mean, stddev, fitted := norm.GetParams()
				if !fitted {
					t.Error("normalizer should be fitted")
				}

				compareFloat := func(got, want float64, name string, index int) {
					if math.IsNaN(want) {
						if !math.IsNaN(got) {
							t.Errorf("%s[%d] = %v; want NaN", name, index, got)
						}
					} else if math.IsInf(want, 0) {
						if !math.IsInf(got, int(math.Copysign(1, want))) {
							t.Errorf("%s[%d] = %v; want %v", name, index, got, want)
						}
					} else {
						if math.Abs(got-want) > 1e-10 {
							t.Errorf("%s[%d] = %v; want %v", name, index, got, want)
						}
					}
				}

				for i := range tc.expectedMean {
					compareFloat(mean[i], tc.expectedMean[i], "mean", i)
				}

				for i := range tc.expectedStddev {
					compareFloat(stddev[i], tc.expectedStddev[i], "stddev", i)
				}
			})
		}
	})
}

func TestNormalizerTransform(t *testing.T) {
	t.Run("successful_transform", func(t *testing.T) {
		t.Parallel()
		norm := NewNormalizer()

		// Fit on known data
		fitData := [][]float64{
			{1.0, 10.0},
			{2.0, 20.0},
			{3.0, 30.0},
		}

		err := norm.Fit(fitData)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// Transform test cases
		tests := []struct {
			name     string
			input    [][]float64
			expected [][]float64
		}{
			{
				name:  "training_data",
				input: fitData,
				expected: [][]float64{
					{-math.Sqrt(3.0 / 2.0), -math.Sqrt(3.0 / 2.0)}, // exactly -sqrt(1.5)
					{0, 0},
					{math.Sqrt(3.0 / 2.0), math.Sqrt(3.0 / 2.0)}, // exactly sqrt(1.5)
				}, // approximate
			},
			{
				name:     "new_data",
				input:    [][]float64{{4.0, 40.0}},
				expected: [][]float64{{math.Sqrt(6), math.Sqrt(6)}}, // (4-2)/sqrt(2/3) = 2/sqrt(2/3) = 2*sqrt(3/2) = sqrt(12/2) = sqrt(6)

				// (4-2)/sqrt(2/3) = 2/sqrt(2/3) = 2*sqrt(3/2) = sqrt(12/2) = sqrt(6)
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				normalized, err := norm.Transform(tc.input)
				if err != nil {
					t.Fatalf("Transform failed: %v", err)
				}

				if len(normalized) != len(tc.input) {
					t.Errorf("normalized length = %d; want %d", len(normalized), len(tc.input))
				}

				if tc.name == "training_data" {
					for feature := 0; feature < 2; feature++ {
						sum := 0.0
						for i := range normalized {
							sum += normalized[i][feature]
						}
						mean := sum / float64(len(normalized))
						if math.Abs(mean) > 1e-10 {
							t.Errorf("Feature %d mean = %v; should be ~0", feature, mean)
						}
					}
				}
			})
		}
	})

	t.Run("error_cases", func(t *testing.T) {
		tests := []struct {
			name      string
			setupFunc func() *Normalizer
			data      [][]float64
		}{
			{
				"not_fitted",
				func() *Normalizer { return NewNormalizer() },
				[][]float64{{1.0, 2.0}},
			},
			{
				"empty_data",
				func() *Normalizer {
					norm := NewNormalizer()
					norm.Fit([][]float64{{1.0, 2.0}})
					return norm
				},
				[][]float64{},
			},
			{
				"feature_number_mismatch",
				func() *Normalizer {
					norm := NewNormalizer()
					norm.Fit([][]float64{{1.0, 2.0}}) // 2 features
					return norm
				},
				[][]float64{{1.0, 2.0, 3.0}}, // 3 features
			},
			{
				"inconsistent_features",
				func() *Normalizer {
					norm := NewNormalizer()
					norm.Fit([][]float64{{1.0, 2.0}}) // 2 features
					return norm
				},
				[][]float64{{1.0, 2.0}, {3.0}}, // Second sample has 1 feature
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				norm := tc.setupFunc()
				_, err := norm.Transform(tc.data)

				if err == nil {
					t.Error("expected error but got nil")
				}
			})
		}
	})
}

func TestFitTransform(t *testing.T) {
	t.Run("equivalent_to_separate_calls", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 10.0},
			{2.0, 20.0},
			{3.0, 30.0},
		}

		// Method 1: FitTransform
		norm1 := NewNormalizer()
		result1, err1 := norm1.FitTransform(data)
		if err1 != nil {
			t.Fatalf("FitTransform failed: %v", err1)
		}

		// Method 2: Separate Fit and Transform
		norm2 := NewNormalizer()
		err2 := norm2.Fit(data)
		if err2 != nil {
			t.Fatalf("Fit failed: %v", err2)
		}
		result2, err3 := norm2.Transform(data)
		if err3 != nil {
			t.Fatalf("Transform failed: %v", err3)
		}

		// Results should be identical
		if len(result1) != len(result2) {
			t.Fatalf("result lengths differ: %d vs %d", len(result1), len(result2))
		}

		for i := range result1 {
			assertFloatSlicesEqual(t, result1[i], result2[i], 1e-10)
		}
	})
}

func TestGetParams(t *testing.T) {
	t.Run("fitted_normalizer", func(t *testing.T) {
		t.Parallel()
		norm := NewNormalizer()
		data := [][]float64{{1.0, 10.0}, {3.0, 30.0}}

		norm.Fit(data)
		mean, stddev, fitted := norm.GetParams()

		if !fitted {
			t.Error("expected fitted to be true")
		}

		expectedMean := []float64{2.0, 20.0}
		expectedStddev := []float64{1.0, 10.0}

		assertFloatSlicesEqual(t, mean, expectedMean, 1e-10)
		assertFloatSlicesEqual(t, stddev, expectedStddev, 1e-10)
	})

	t.Run("returns_copies", func(t *testing.T) {
		t.Parallel()
		norm := NewNormalizer()
		data := [][]float64{{1.0, 10.0}, {3.0, 30.0}}
		norm.Fit(data)

		mean, stddev, _ := norm.GetParams()

		// Modify the returned slices
		mean[0] = 999
		stddev[0] = 999

		// Original normalizer should be unchanged
		if norm.mean[0] == 999 || norm.stddev[0] == 999 {
			t.Error("GetParams should return a copy of mean, not a reference")
		}
	})

	t.Run("unfitted_normalizer", func(t *testing.T) {
		t.Parallel()

		norm := NewNormalizer()
		mean, stddev, fitted := norm.GetParams()

		if fitted {
			t.Error("expected fitted to be false for new normalizer")
		}

		if mean != nil {
			t.Error("expected mean to be nil for unfitted normalizer")
		}

		if stddev != nil {
			t.Error("expected stddev to be nil for unfitted normalizer")
		}
	})
}

// =============================================================================
// PROPERTY TESTS
// =============================================================================

func TestNormalizerProperties(t *testing.T) {
	t.Run("normalization_properties", func(t *testing.T) {
		t.Run("normalized_data_with_zero_mean_unit_variance", func(t *testing.T) {
			t.Parallel()
			// Property: After normalization, each feature should have mean ≈ 0 and stddev ≈ 1
			testSizes := []struct {
				samples, features int
			}{
				{10, 2}, {50, 3}, {100, 5},
			}

			for _, size := range testSizes {
				norm := NewNormalizer()
				data := generateRandomData(size.samples, size.features)

				normalized, err := norm.FitTransform(data)
				if err != nil {
					t.Errorf("FitTransform failed for size %dx%d: %v", size.samples, size.features, err)
					continue
				}

				for feature := 0; feature < size.features; feature++ {
					sum := 0.0
					for sample := 0; sample < size.samples; sample++ {
						sum += normalized[sample][feature]
					}
					mean := sum / float64(size.samples)

					sumSquares := 0.0
					for sample := 0; sample < size.samples; sample++ {
						diff := normalized[sample][feature] - mean
						sumSquares += diff * diff
					}
					variance := sumSquares / float64(size.samples)
					stddev := math.Sqrt(variance)

					if math.Abs(mean) > 1e-10 {
						t.Errorf("Feature %d mean = %v; should be ≈ 0 for size %dx%d", feature, mean, size.samples, size.features)
					}
					if math.Abs(stddev-1.0) > 1e-10 {
						t.Errorf("Feature %d stddev = %v; should be ≈ 1.0", feature, stddev)
					}
				}
			}
		})

		t.Run("identity_property", func(t *testing.T) {
			t.Parallel()
			// Property: Transforming the training data should give the same result as FitTransform

			data := generateRandomData(50, 4)

			// Method 1: FitTransform
			norm1 := NewNormalizer()
			result1, _ := norm1.FitTransform(data)

			// Method 2: Fit then Transform the same data
			norm2 := NewNormalizer()
			norm2.Fit(data)
			result2, _ := norm2.Transform(data)

			// Results should be identical
			for i := range result1 {
				for j := range result1[i] {
					if math.Abs(result1[i][j]-result2[i][j]) > 1e-15 {
						t.Errorf("Identity property failed at [%d][%d]: %v vs %v", i, j, result1[i][j], result2[i][j])
					}
				}
			}
		})

		t.Run("consistency_property", func(t *testing.T) {
			t.Parallel()
			// Property: Same input should always produce same output

			norm := NewNormalizer()
			data := generateRandomData(30, 2)
			norm.Fit(data)

			testInput := [][]float64{{1.5, 2.5}, {3.0, 4.0}}

			result1, _ := norm.Transform(testInput)
			result2, _ := norm.Transform(testInput)

			for i := range result1 {
				for j := range result1[i] {
					if result1[i][j] != result2[i][j] {
						t.Errorf("Consistency property failed: same input produced different outputs")
					}
				}
			}
		})

		t.Run("affine_transformation_property", func(t *testing.T) {
			t.Parallel()
			// Property: Normalization is an affine transformation: f(x) = (x - mean) / stddev

			norm := NewNormalizer()
			data := generateRandomData(50, 2)
			norm.Fit(data)

			// Test that the transformation is mathematically correct
			x := [][]float64{{1.0, 2.0}}
			y, _ := norm.Transform(x)

			// Verify the transformation
			for j := 0; j < 2; j++ {
				expected := (x[0][j] - norm.mean[j]) / norm.stddev[j]
				if math.Abs(y[0][j]-expected) > 1e-10 {
					t.Errorf("Affine transformation failed for feature %d: got %v, expected %v",
						j, y[0][j], expected)
				}
			}
		})
	})
}

// ============================================================================
// ROBUSTNESS TESTS
// ============================================================================

func TestNormalizerRobustness(t *testing.T) {
	t.Run("handles_extreme_values", func(t *testing.T) {
		t.Parallel()
		// Normalizer should handle very large and very small values

		data := [][]float64{
			{1e-10, 1e10},
			{2e-10, 2e10},
			{3e-10, 3e10},
		}

		norm := NewNormalizer()
		normalized, err := norm.FitTransform(data)
		if err != nil {
			t.Errorf("Failed to handle extreme values: %v", err)
		}

		// Check that results are finite
		for i := range normalized {
			for j := range normalized[i] {
				if math.IsNaN(normalized[i][j]) || math.IsInf(normalized[i][j], 0) {
					t.Errorf("Extreme values produced non-finite result: %v", normalized[i][j])
				}
			}
		}
	})

	t.Run("handles_zero_variance_gracefully", func(t *testing.T) {
		t.Parallel()
		// Constant features should be handled without errors

		data := [][]float64{
			{5.0, 1.0}, // Second feature varies
			{5.0, 2.0}, // First feature is constant
			{5.0, 3.0},
		}

		norm := NewNormalizer()
		normalized, err := norm.FitTransform(data)
		if err != nil {
			t.Errorf("Failed to handle constant feature: %v", err)
		}

		// Constant feature should normalize to zero (since mean is subtracted and stddev=1)
		for i := range normalized {
			if math.Abs(normalized[i][0]) > 1e-10 {
				t.Errorf("Constant feature not handled correctly: got %v, want 0", normalized[i][0])
			}
		}

		_, stddev, _ := norm.GetParams()
		if stddev[0] != 1.0 {
			t.Errorf("Constant feature stddev = %v; should be 1.0", stddev[0])
		}
	})

	t.Run("numerical_stability", func(t *testing.T) {
		t.Parallel()
		// Test with values that might cause numerical issues

		tests := []struct {
			name string
			data [][]float64
		}{
			{
				name: "very_small_variance",
				data: [][]float64{
					{1.0, 1.0 + 1e-15},
					{1.0, 1.0 + 2e-15},
					{1.0, 1.0 + 3e-15},
				},
			},
			{
				name: "large_magnitude_small_variance",
				data: [][]float64{
					{1e10, 1e10 + 1},
					{1e10, 1e10 + 2},
					{1e10, 1e10 + 3},
				},
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				norm := NewNormalizer()
				normalized, err := norm.FitTransform(tc.data)
				if err != nil {
					t.Errorf("Failed on %s: %v", tc.name, err)
				}

				// Results should be finite
				for i := range normalized {
					for j := range normalized[i] {
						if math.IsNaN(normalized[i][j]) || math.IsInf(normalized[i][j], 0) {
							t.Errorf("%s produced non-finite result: %v", tc.name, normalized[i][j])
						}
					}
				}
			})
		}
	})
}

// =============================================================================
// BENCHMARK TESTS
// =============================================================================

func BenchmarkNormalizerFit(b *testing.B) {
	b.Run("samples_scaling", func(b *testing.B) {
		features := 10
		sampleSizes := []int{100, 1000, 10000}

		for _, samples := range sampleSizes {
			b.Run(fmt.Sprintf("%dx%d", samples, features), func(b *testing.B) {
				data := generateRandomData(samples, features)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					norm := NewNormalizer()
					norm.Fit(data)
				}
			})
		}
	})

	b.Run("features_scaling", func(b *testing.B) {
		samples := 1000
		featureSizes := []int{10, 100, 10000}

		for _, features := range featureSizes {
			b.Run(fmt.Sprintf("%dx%d", samples, features), func(b *testing.B) {
				data := generateRandomData(samples, features)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					norm := NewNormalizer()
					norm.Fit(data)
				}
			})
		}
	})
}

func BenchmarkNormalizerTransform(b *testing.B) {
	b.Run("samples_scaling", func(b *testing.B) {
		features := 10
		sampleSizes := []int{100, 1000, 10000}

		for _, samples := range sampleSizes {
			b.Run(fmt.Sprintf("%dx%d", samples, features), func(b *testing.B) {
				trainData := generateRandomData(samples, features)
				testData := generateRandomData(samples, features)

				norm := NewNormalizer()
				norm.Fit(trainData)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					_, _ = norm.Transform(testData)
				}
			})
		}
	})

	b.Run("features_scaling", func(b *testing.B) {
		samples := 1000
		featureSizes := []int{10, 100, 1000, 10000}

		for _, features := range featureSizes {
			b.Run(fmt.Sprintf("%dx%d", samples, features), func(b *testing.B) {
				trainData := generateRandomData(samples, features)
				testData := generateRandomData(samples, features)

				norm := NewNormalizer()
				norm.Fit(trainData)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					_, _ = norm.Transform(testData)
				}
			})
		}
	})
}

func BenchmarkNormalizerFitTransform(b *testing.B) {
	dataSizes := []struct {
		samples, features int
	}{
		{100, 10},   // Small
		{1000, 10},  // Medium
		{10000, 10}, // Large

		{1000, 100},  // Wide
		{1000, 1000}, // Very Wide
	}

	for _, size := range dataSizes {
		b.Run(fmt.Sprintf("%dx%d", size.samples, size.features), func(b *testing.B) {
			data := generateRandomData(size.samples, size.features)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				norm := NewNormalizer()
				_, _ = norm.FitTransform(data)
			}
		})
	}
}
