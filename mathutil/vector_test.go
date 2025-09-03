package mathutil

import (
	"math"
	"math/rand/v2"
	"testing"
)

// ============================================================================
// TEST HELPERS
// ============================================================================

func createTestVectors(size int) ([]float64, []float64) {
	a := make([]float64, size)
	b := make([]float64, size)
	for i := range a {
		a[i] = float64(i) * 0.1
		b[i] = float64(i) * 0.2
	}
	return a, b
}

// ============================================================================
// UNIT TESTS
// ============================================================================

func TestDotProduct(t *testing.T) {
	t.Run("basic_cases", func(t *testing.T) {
		t.Parallel()
		tests := []struct {
			name string
			a, b []float64
			want float64
		}{
			{"simple_positive", []float64{1, 2, 3}, []float64{4, 5, 6}, 32}, // 1*4 + 2*5 + 3*6 = 32
			{"zero_vectors", []float64{0, 0, 0}, []float64{1, 2, 3}, 0},
			{"mixed_signs", []float64{1.5, -2.0}, []float64{2.0, 3.0}, -3.0}, // 3.0 - 6.0 = -3.0
			{"empty_vectors", []float64{}, []float64{}, 0},
			{"single_element", []float64{5}, []float64{7}, 35},
			{"all_negative", []float64{-1, -2}, []float64{-3, -4}, 11}, // 3 + 8 = 11
			{"orthogonal", []float64{1, 0}, []float64{0, 1}, 0},
			{"identical_vectors", []float64{2, 3, 4}, []float64{2, 3, 4}, 29}, // 4+9+16=29
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				got := DotProduct(tc.a, tc.b)

				if got != tc.want {
					t.Errorf("DotProduct(%v, %v) = %v; want %v", tc.a, tc.b, got, tc.want)
				}
			})
		}
	})

	t.Run("special_float_values", func(t *testing.T) {
		t.Parallel()
		tests := []struct {
			name    string
			a, b    []float64
			want    float64
			wantNaN bool
		}{
			{"with_infinity", []float64{1, math.Inf(1)}, []float64{1, 1}, math.Inf(1), false},
			{"with_negative_infinity", []float64{1, math.Inf(-1)}, []float64{1, 1}, math.Inf(-1), false},
			{"infinity_times_zero", []float64{math.Inf(1), 0}, []float64{0, 1}, math.NaN(), true},
			{"nan_propagation", []float64{math.NaN(), 1}, []float64{1, 1}, math.NaN(), true},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				got := DotProduct(tc.a, tc.b)

				if tc.wantNaN {
					if !math.IsNaN(got) {
						t.Errorf("DotProduct(%v, %v) = %v; want NaN", tc.a, tc.b, got)
					}
				} else if got != tc.want {
					t.Errorf("DotProduct(%v, %v) = %v; want %v", tc.a, tc.b, got, tc.want)
				}
			})
		}
	})

	t.Run("panic_on_length_mismatch", func(t *testing.T) {
		tests := []struct {
			name string
			a, b []float64
		}{
			{"different_lengths", []float64{1, 2}, []float64{1}},
			{"first_longer", []float64{1, 2, 3}, []float64{1, 2}},
			{"second_longer", []float64{1, 2}, []float64{1, 2, 3}},
			{"empty_vs_nonempty", []float64{}, []float64{1}},
			{"nonempty_vs_empty", []float64{1}, []float64{}},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("DotProduct(%v, %v) should panic on length mismatch", tc.a, tc.b)
					}
				}()
				DotProduct(tc.a, tc.b)
			})
		}
	})
}

// ============================================================================
// PROPERTY TESTS
// ============================================================================

func TestDotProductProperties(t *testing.T) {
	t.Run("commutativity", func(t *testing.T) {
		t.Parallel()
		// Property: a*b = b*a
		testPairs := []struct{ a, b []float64 }{
			{[]float64{1, 2, 3}, []float64{4, 5, 6}},
			{[]float64{-1, 0, 2.5}, []float64{3, -7, 1.2}},
			{[]float64{0, 0}, []float64{1, 1}},
			{[]float64{math.Pi, math.E}, []float64{1, -1}},
		}

		for _, pair := range testPairs {
			res1 := DotProduct(pair.a, pair.b)
			res2 := DotProduct(pair.b, pair.a)

			if res1 != res2 {
				t.Errorf("Commutativity failed: %v·%v = %v, %v·%v = %v", pair.a, pair.b, res1, pair.b, pair.a, res2)
			}
		}
	})

	t.Run("zero_vector", func(t *testing.T) {
		t.Parallel()
		// Property: zero vector dot product with any vector is zero
		testVectors := [][]float64{
			{1, 2, 3},
			{-5, 10, -2.5},
			{math.Pi, math.E, math.Sqrt2},
			{1e6, -1e6, 0},
		}
		for _, vec := range testVectors {
			zero := make([]float64, len(vec))
			result := DotProduct(zero, vec)
			if result != 0 {
				t.Errorf("Zero vector property failed: 0·%v = %v; want 0", vec, result)
			}
		}
	})
}

// ============================================================================
// ROBUSTNESS TESTS
// ============================================================================

func TestDotProductRobustness(t *testing.T) {
	t.Run("overflow_detection", func(t *testing.T) {
		t.Parallel()
		// Test potential overflow
		a := []float64{math.MaxFloat64 / 2, math.MaxFloat64 / 2}
		b := []float64{2, 2}

		result := DotProduct(a, b)

		// Should be infinity due to overflow
		if !math.IsInf(result, 1) {
			t.Errorf("Expected positive infinity for overflow case, got %v", result)
		}
	})
}

// ============================================================================
// BENCHMARKS
// ============================================================================

func BenchmarkDotProduct(b *testing.B) {
	vectorSizes := []struct {
		name string
		size int
	}{
		{"tiny", 2},
		{"small", 10},
		{"medium", 100},
		{"large", 1000},
		{"very_large", 10000},
	}

	for _, vs := range vectorSizes {
		a, vecB := createTestVectors(vs.size)

		b.Run(vs.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				DotProduct(a, vecB)
			}
		})
	}
}

func BenchmarkDotProductMemory(b *testing.B) {
	a := []float64{1, 2, 3, 4, 5}
	vecB := []float64{6, 7, 8, 9, 10}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DotProduct(a, vecB)
	}
}

func BenchmarkDotProductNeuralNetwork(b *testing.B) {
	size := 1000

	b.Run("neural_network_specific", func(b *testing.B) {
		a := make([]float64, size)
		vecB := make([]float64, size)
		for i := range a {
			a[i] = rand.NormFloat64() * 0.1
			vecB[i] = rand.NormFloat64() * 0.1
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProduct(a, vecB)
		}
	})
}
