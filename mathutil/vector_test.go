package mathutil

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestDotProduct(t *testing.T) {
	t.Run("basic_cases", func(t *testing.T) {
		tests := []struct {
			name    string
			a, b    []float64
			want    float64
			wantNaN bool
		}{
			{"simple_positive", []float64{1, 2, 3}, []float64{4, 5, 6}, 32, false}, // 1*4 + 2*5 + 3*6 = 32
			{"zero_vectors", []float64{0, 0, 0}, []float64{1, 2, 3}, 0, false},
			{"mixed_signs", []float64{1.5, -2.0}, []float64{2.0, 3.0}, -3.0, false}, // 3.0 - 6.0 = -3.0
			{"empty_vectors", []float64{}, []float64{}, 0, false},
			{"single_element", []float64{5}, []float64{7}, 35, false},
			{"all_negative", []float64{-1, -2}, []float64{-3, -4}, 11, false}, // 3 + 8 = 11
			{"orthogonal", []float64{1, 0}, []float64{0, 1}, 0, false},
			{"identical_vectors", []float64{2, 3, 4}, []float64{2, 3, 4}, 29, false}, // 4+9+16=29
			{"with_infinity", []float64{1, math.Inf(1)}, []float64{1, 1}, math.Inf(1), false},
			{"with_negative_infinity", []float64{1, math.Inf(-1)}, []float64{1, 1}, math.Inf(-1), false},
			{"infinity_times_zero", []float64{math.Inf(1), 0}, []float64{0, 1}, math.NaN(), true},
			{"nan_propagation", []float64{math.NaN(), 1}, []float64{1, 1}, math.NaN(), true},
			{"negative_zero", []float64{-0.0}, []float64{1}, 0, false},
			{"very_small_numbers", []float64{1e-100, 1e-100}, []float64{1e-100, 1e-100}, 2e-200, false},
			{"very_large_numbers", []float64{1e100, 1e100}, []float64{1e100, 1e100}, 2e200, false},
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

	t.Run("properties", func(t *testing.T) {
		t.Parallel()

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

		t.Run("distributivity", func(t *testing.T) {
			t.Parallel()
			// Property: a*(b+c) = a*b+a*c
			a := []float64{1, 2, 3}
			b := []float64{2, -1, 4}
			c := []float64{0, 3, -2}

			bPlusC := make([]float64, len(b))
			for i := range b {
				bPlusC[i] = b[i] + c[i]
			}
			left := DotProduct(a, bPlusC)
			right := DotProduct(a, b) + DotProduct(a, c)

			if math.Abs(left-right) > 1e-10 {
				t.Errorf("Distributivity failed: a·(b+c) = %v, a·b + a·c = %v", left, right)
			}
		})

		t.Run("scalar_multiplication", func(t *testing.T) {
			t.Parallel()
			// Property: (k*a)*b = k*(a*b)
			a := []float64{1, 2, 3}
			b := []float64{4, 5, 6}
			k := 2.5

			kTimesA := make([]float64, len(a))
			for i := range a {
				kTimesA[i] = k * a[i]
			}

			left := DotProduct(kTimesA, b)
			right := k * DotProduct(a, b)

			if math.Abs(left-right) > 1e-10 {
				t.Errorf("Scalar multiplication failed: (k*a)·b = %v, k*(a·b) = %v", left, right)
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

		t.Run("unit_vector_magnitude", func(t *testing.T) {
			t.Parallel()
			// Property: unit vector dot product with itself equals 1
			unitVectors := [][]float64{
				{1},                                  // 1D unit vector
				{1, 0},                               // 2D unit vector along x
				{0, 1},                               // 2D unit vector along y
				{1 / math.Sqrt(2), 1 / math.Sqrt(2)}, // 2D unit vector at 45 degrees
				{1 / math.Sqrt(3), 1 / math.Sqrt(3), 1 / math.Sqrt(3)}, // 3D unit vector
			}

			for _, unit := range unitVectors {
				result := DotProduct(unit, unit)
				if math.Abs(result-1.0) > 1e-10 {
					t.Errorf("Unit vector magnitude failed: %v·%v = %v; want 1.0", unit, unit, result)
				}
			}
		})

		t.Run("orthogonal_vectors", func(t *testing.T) {
			t.Parallel()
			// Property: orthogonal vectors have zero dot product
			orthogonalPairs := []struct{ a, b []float64 }{
				{[]float64{1, 0}, []float64{0, 1}},
				{[]float64{1, 0, 0}, []float64{0, 1, 0}},
				{[]float64{1, 0, 0}, []float64{0, 0, 1}},
				{[]float64{3, 4}, []float64{4, -3}}, // perpendicular vectors
			}

			for _, pair := range orthogonalPairs {
				result := DotProduct(pair.a, pair.b)

				if math.Abs(result) > 1e-10 {
					t.Errorf("Orthogonal vectors failed: %v·%v = %v; want 0", pair.a, pair.b, result)
				}
			}
		})

		t.Run("linearity", func(t *testing.T) {
			t.Parallel()
			// Property: (a+b)*c = a*c + b*c
			a := []float64{1, 2}
			b := []float64{3, 4}
			c := []float64{5, 6}

			aPlusB := make([]float64, len(a))
			for i := range a {
				aPlusB[i] = a[i] + b[i]
			}
			left := DotProduct(aPlusB, c)
			right := DotProduct(a, c) + DotProduct(b, c)

			if math.Abs(left-right) > 1e-10 {
				t.Errorf("Linearity failed: (a+b)·c = %v, a·c + b·c = %v", left, right)
			}
		})
	})
}

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
		// create test vectors
		a := make([]float64, vs.size)
		vecB := make([]float64, vs.size)

		for i := range a {
			a[i] = float64(i) * 0.1
			vecB[i] = float64(i) * 0.2
		}

		b.Run(vs.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				DotProduct(a, vecB)
			}
		})
	}
}

func BenchmarkDotProductMemoryAllocation(b *testing.B) {
	a := []float64{1, 2, 3, 4, 5}
	vecB := []float64{6, 7, 8, 9, 10}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DotProduct(a, vecB)
	}
}

func BenchmarkDotProductSpecialCases(b *testing.B) {
	scenarios := []struct {
		name    string
		a, vecB []float64
	}{
		{"zero_vectors", make([]float64, 1000), make([]float64, 1000)},
		{"unit_vectors", []float64{1, 0, 0, 0, 0}, []float64{0, 1, 0, 0, 0}},
		{"large_numbers", []float64{1e6, 1e6, 1e6}, []float64{1e6, 1e6, 1e6}},
		{"small_numbers", []float64{1e-6, 1e-6, 1e-6}, []float64{1e-6, 1e-6, 1e-6}},
	}

	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				DotProduct(scenario.a, scenario.vecB)
			}
		})
	}
}

func BenchmarkDotProductVectorTypes(b *testing.B) {
	size := 1000

	b.Run("sequential", func(b *testing.B) {
		a := make([]float64, size)
		vecB := make([]float64, size)

		for i := range a {
			a[i] = float64(i)
			vecB[i] = float64(i)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProduct(a, vecB)
		}
	})

	b.Run("random_like", func(b *testing.B) {
		a := make([]float64, size)
		vecB := make([]float64, size)

		for i := range a {
			a[i] = float64(i%7) * 0.1
			vecB[i] = float64((i*3)%11) * 0.2
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProduct(a, vecB)
		}
	})

	b.Run("alternating_signs", func(b *testing.B) {
		a := make([]float64, size)
		vecB := make([]float64, size)

		for i := range a {
			sign := 1.0
			if i%2 == 1 {
				sign = -1.0
			}
			a[i] = sign * float64(i)
			vecB[i] = sign * float64(i)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProduct(a, vecB)
		}
	})

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
