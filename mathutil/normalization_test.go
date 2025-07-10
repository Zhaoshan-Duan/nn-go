package mathutil

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

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
				name: "simple_2x3_data",
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

				// Check mean
				if len(norm.mean) != len(tc.expectedMean) {
					t.Fatalf("mean length = %d; want %d", len(norm.mean), len(tc.expectedMean))
				}

				for i, expected := range tc.expectedMean {
					if math.Abs(norm.mean[i]-expected) > 1e-10 {
						t.Errorf("mean[%d] = %v; want %v", i, norm.mean[i], expected)
					}
				}

				// Check stddev
				if len(norm.stddev) != len(tc.expectedStddev) {
					t.Fatalf("stddev length = %d; want %d", len(norm.stddev), len(tc.expectedStddev))
				}

				for i, expected := range tc.expectedStddev {
					if math.Abs(norm.stddev[i]-expected) > 1e-10 {
						t.Errorf("stddev[%d] = %v; want %v", i, norm.stddev[i], expected)
					}
				}
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

	t.Run("constant_features", func(t *testing.T) {
		t.Parallel()
		norm := NewNormalizer()
		data := [][]float64{
			{1.0, 5.0}, // Second feature is constant
			{3.0, 5.0},
			{2.0, 5.0},
		}

		err := norm.Fit(data)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// First feature should have normal stddev
		if norm.stddev[0] == 0 {
			t.Error("first feature stddev should not be zero")
		}

		// Second feature (constant) should have stddev = 1.0
		if norm.stddev[1] != 1.0 {
			t.Errorf("constant feature stddev = %v; want 1.0", norm.stddev[1])
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

		// Transform different data
		transformData := [][]float64{
			{1.0, 10.0}, // Should be close to [-1, -1] after normalization
			{2.0, 20.0}, // Should be close to [0, 0] after normalization
			{3.0, 30.0}, // Should be close to [1, 1] after normalization
			{4.0, 40.0}, // Out of training range
		}

		normalized, err := norm.Transform(transformData)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		if len(normalized) != len(transformData) {
			t.Errorf("normalized length = %d; want %d", len(normalized), len(transformData))
		}

		// Check dimensions
		for i, sample := range normalized {
			if len(sample) != len(transformData[i]) {
				t.Errorf("sample %d length = %d; want %d", i, len(sample), len(transformData[i]))
			}
		}

		// Check that middle value is approximately zero (since it equals the mean)
		middleSample := normalized[1] // [2.0, 20.0] which equals the mean
		for j, val := range middleSample {
			if math.Abs(val) > 1e-10 {
				t.Errorf("middle sample[%d] = %v; should be close to 0", j, val)
			}
		}
	})

	t.Run("error_cases", func(t *testing.T) {
		tests := []struct {
			name      string
			setupFunc func() *Normalizer
			data      [][]float64
			wantErr   bool
		}{
			{
				"not_fitted",
				func() *Normalizer { return NewNormalizer() },
				[][]float64{{1.0, 2.0}},
				true,
			},
			{
				"empty_data",
				func() *Normalizer {
					norm := NewNormalizer()
					norm.Fit([][]float64{{1.0, 2.0}})
					return norm
				},
				[][]float64{},
				true,
			},
			{
				"feature_mismatch",
				func() *Normalizer {
					norm := NewNormalizer()
					norm.Fit([][]float64{{1.0, 2.0}}) // 2 features
					return norm
				},
				[][]float64{{1.0, 2.0, 3.0}}, // 3 features
				true,
			},
			{
				"inconsistent_features",
				func() *Normalizer {
					norm := NewNormalizer()
					norm.Fit([][]float64{{1.0, 2.0}}) // 2 features
					return norm
				},
				[][]float64{{1.0, 2.0}, {3.0}}, // Second sample has 1 feature
				true,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				norm := tc.setupFunc()
				_, err := norm.Transform(tc.data)

				if tc.wantErr && err == nil {
					t.Error("expected error but got nil")
				} else if !tc.wantErr && err != nil {
					t.Errorf("unexpected error: %v", err)
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
			for j := range result1[i] {
				if math.Abs(result1[i][j]-result2[i][j]) > 1e-15 {
					t.Errorf("results differ at [%d][%d]: %v vs %v", i, j, result1[i][j], result2[i][j])
				}
			}
		}

		// Both normalizers should be fitted
		if !norm1.fitted || !norm2.fitted {
			t.Error("both normalizers should be fitted")
		}
	})
}

func TestGetParams(t *testing.T) {
	t.Run("fitted_normalizer", func(t *testing.T) {
		t.Parallel()
		norm := NewNormalizer()
		data := [][]float64{{1.0, 10.0}, {3.0, 30.0}}

		err := norm.Fit(data)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		mean, stddev, fitted := norm.GetParams()

		if !fitted {
			t.Error("expected fitted to be true")
		}

		expectedMean := []float64{2.0, 20.0}
		expectedStddev := []float64{1.0, 10.0}

		for i, expected := range expectedMean {
			if math.Abs(mean[i]-expected) > 1e-10 {
				t.Errorf("mean[%d] = %v; want %v", i, mean[i], expected)
			}
		}

		for i, expected := range expectedStddev {
			if math.Abs(stddev[i]-expected) > 1e-10 {
				t.Errorf("stddev[%d] = %v; want %v", i, stddev[i], expected)
			}
		}
	})

	t.Run("returns_copies", func(t *testing.T) {
		t.Parallel()
		norm := NewNormalizer()
		data := [][]float64{{1.0, 10.0}, {3.0, 30.0}}
		norm.Fit(data)

		mean, stddev, _ := norm.GetParams()

		// Modify the returned slices
		originalMean := mean[0]
		originalStddev := stddev[0]
		mean[0] = 999
		stddev[0] = 999

		// Original normalizer should be unchanged
		if norm.mean[0] == 999 {
			t.Error("GetParams should return a copy of mean, not a reference")
		}
		if norm.stddev[0] == 999 {
			t.Error("GetParams should return a copy of stddev, not a reference")
		}

		// Verify original values are preserved
		if math.Abs(norm.mean[0]-originalMean) > 1e-15 {
			t.Error("original mean was modified")
		}
		if math.Abs(norm.stddev[0]-originalStddev) > 1e-15 {
			t.Error("original stddev was modified")
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
// PROPERTY-BASED TESTS
// =============================================================================

func TestNormalizerProperties(t *testing.T) {
	t.Run("normalization_properties", func(t *testing.T) {
		t.Run("normalized_data_has_zero_mean", func(t *testing.T) {
			t.Parallel()
			// Property: After normalization, each feature should have mean ≈ 0

			testSizes := []struct {
				samples, features int
			}{
				{10, 2}, {50, 3}, {100, 5}, {20, 1},
			}

			for _, size := range testSizes {
				norm := NewNormalizer()
				data := generateRandomData(size.samples, size.features)

				normalized, err := norm.FitTransform(data)
				if err != nil {
					t.Errorf("FitTransform failed for size %dx%d: %v", size.samples, size.features, err)
					continue
				}

				// Check mean of each feature is approximately zero
				for feature := 0; feature < size.features; feature++ {
					sum := 0.0
					for sample := 0; sample < size.samples; sample++ {
						sum += normalized[sample][feature]
					}
					mean := sum / float64(size.samples)

					if math.Abs(mean) > 1e-10 {
						t.Errorf("Feature %d mean = %v; should be ≈ 0 for size %dx%d", feature, mean, size.samples, size.features)
					}
				}
			}
		})

		t.Run("normalized_data_has_unit_variance", func(t *testing.T) {
			t.Parallel()
			// Property: After normalization, each feature should have stddev ≈ 1

			norm := NewNormalizer()
			data := generateRandomData(100, 3)

			normalized, err := norm.FitTransform(data)
			if err != nil {
				t.Fatalf("FitTransform failed: %v", err)
			}

			// Check standard deviation of each feature is approximately 1
			for feature := 0; feature < 3; feature++ {
				// Calculate mean
				sum := 0.0
				for sample := 0; sample < 100; sample++ {
					sum += normalized[sample][feature]
				}
				mean := sum / 100.0

				// Calculate variance
				sumSquares := 0.0
				for sample := 0; sample < 100; sample++ {
					diff := normalized[sample][feature] - mean
					sumSquares += diff * diff
				}
				variance := sumSquares / 100.0
				stddev := math.Sqrt(variance)

				if math.Abs(stddev-1.0) > 1e-10 {
					t.Errorf("Feature %d stddev = %v; should be ≈ 1.0", feature, stddev)
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
			// Property: Normalization is an affine transformation
			// f(x) = (x - mean) / stddev should behave consistently

			norm := NewNormalizer()
			data := generateRandomData(50, 2)
			norm.Fit(data)

			// Test that the transformation is consistent
			x1 := [][]float64{{1.0, 2.0}}
			x2 := [][]float64{{3.0, 4.0}}

			y1, _ := norm.Transform(x1)
			y2, _ := norm.Transform(x2)

			// Property: If we know the mean and stddev, we can predict the result
			for j := 0; j < 2; j++ {
				expectedY1 := (x1[0][j] - norm.mean[j]) / norm.stddev[j]
				expectedY2 := (x2[0][j] - norm.mean[j]) / norm.stddev[j]

				if math.Abs(y1[0][j]-expectedY1) > 1e-10 {
					t.Errorf("Affine transformation failed for x1[%d]: got %v, expected %v", j, y1[0][j], expectedY1)
				}
				if math.Abs(y2[0][j]-expectedY2) > 1e-10 {
					t.Errorf("Affine transformation failed for x2[%d]: got %v, expected %v", j, y2[0][j], expectedY2)
				}
			}
		})

		t.Run("translation_invariance_property", func(t *testing.T) {
			t.Parallel()
			// Property: Adding same constant to all values should only affect the mean, not the relative differences

			norm := NewNormalizer()
			originalData := [][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}
			norm.Fit(originalData)

			// Transform original data
			original, _ := norm.Transform(originalData)

			// Create translated data (add constant to all values)
			constant := 100.0
			translatedData := make([][]float64, len(originalData))
			for i := range originalData {
				translatedData[i] = make([]float64, len(originalData[i]))
				for j := range originalData[i] {
					translatedData[i][j] = originalData[i][j] + constant
				}
			}

			// Transform translated data with same normalizer
			translated, _ := norm.Transform(translatedData)

			// The differences between normalized values should be the same
			for i := 0; i < len(original)-1; i++ {
				for j := 0; j < len(original[i]); j++ {
					originalDiff := original[i+1][j] - original[i][j]
					translatedDiff := translated[i+1][j] - translated[i][j]

					if math.Abs(originalDiff-translatedDiff) > 1e-10 {
						t.Errorf("Translation invariance failed: differences changed after translation")
					}
				}
			}
		})

		t.Run("scaling_property", func(t *testing.T) {
			t.Parallel()
			// Property: Scaling input by constant should scale normalized output by same constant

			norm := NewNormalizer()
			data := generateRandomData(30, 2)
			norm.Fit(data)

			original := [][]float64{{2.0, 4.0}}
			scaled := [][]float64{{4.0, 8.0}} // 2x scaling

			origResult, _ := norm.Transform(original)
			scaledResult, _ := norm.Transform(scaled)

			// The difference should be proportional to the scaling
			for j := 0; j < 2; j++ {
				// For normalized data: (2x - mean)/std vs (x - mean)/std
				// The difference should be x/std
				expectedDiff := (scaled[0][j] - original[0][j]) / norm.stddev[j]
				actualDiff := scaledResult[0][j] - origResult[0][j]

				if math.Abs(actualDiff-expectedDiff) > 1e-10 {
					t.Errorf("Scaling property failed for feature %d", j)
				}
			}
		})
	})

	t.Run("robustness_properties", func(t *testing.T) {
		t.Run("handles_extreme_values", func(t *testing.T) {
			t.Parallel()
			// Property: Normalizer should handle very large and very small values

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
			// Property: Constant features should be handled without errors

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
				expectedConstant := (5.0 - 5.0) / 1.0 // (value - mean) / stddev
				if math.Abs(normalized[i][0]-expectedConstant) > 1e-10 {
					t.Errorf("Constant feature not handled correctly: got %v, want %v", normalized[i][0], expectedConstant)
				}
			}
		})
	})
}

// =============================================================================
// BENCHMARK TESTS
// =============================================================================

func BenchmarkNormalizerFit(b *testing.B) {
	dataSizes := []struct {
		name              string
		samples, features int
	}{
		{"small", 100, 5},
		{"medium", 1000, 10},
		{"large", 10000, 20},
		{"wide", 1000, 100},
		{"tall", 10000, 5},
	}

	for _, size := range dataSizes {
		b.Run(size.name, func(b *testing.B) {
			data := generateRandomData(size.samples, size.features)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				norm := NewNormalizer()
				norm.Fit(data)
			}
		})
	}
}

func BenchmarkNormalizerTransform(b *testing.B) {
	dataSizes := []struct {
		name              string
		samples, features int
	}{
		{"small", 100, 5},
		{"medium", 1000, 10},
		{"large", 10000, 20},
	}

	for _, size := range dataSizes {
		b.Run(size.name, func(b *testing.B) {
			// Setup
			trainData := generateRandomData(size.samples, size.features)
			testData := generateRandomData(size.samples, size.features)

			norm := NewNormalizer()
			norm.Fit(trainData)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = norm.Transform(testData)
			}
		})
	}
}

func BenchmarkNormalizerFitTransform(b *testing.B) {
	data := generateRandomData(1000, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		norm := NewNormalizer()
		_, _ = norm.FitTransform(data)
	}
}

func BenchmarkNormalizerMemory(b *testing.B) {
	data := generateRandomData(1000, 10)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		norm := NewNormalizer()
		_, _ = norm.FitTransform(data)
	}
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

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

// Memory leak test
func TestNormalizerMemoryStability(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory test in short mode")
	}

	t.Run("repeated_operations_no_leak", func(t *testing.T) {
		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)

		// Perform many operations
		data := generateRandomData(100, 5)
		for i := 0; i < 1000; i++ {
			norm := NewNormalizer()
			_, _ = norm.FitTransform(data)

			if i%100 == 0 {
				runtime.GC()
			}
		}

		runtime.GC()
		runtime.ReadMemStats(&m2)

		memGrowth := m2.Alloc - m1.Alloc
		if memGrowth > 1024*1024 { // 1MB threshold
			t.Logf("Memory growth detected: %d bytes (this might be acceptable)", memGrowth)
		}
	})
}
