package activation

import (
	"math"
	"testing"
)

// ============================================================================
// TEST HELPERS
// ============================================================================

func assertFloat64Equal(t *testing.T, got, want, tolerance float64, name string) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Errorf("%s: got %v, want %v", name, got, want)
	}
}

func getActivationNames() []string {
	return []string{"relu", "sigmoid", "tanh", "linear"}
}

// ============================================================================
// UNIT TESTS
// ============================================================================

func TestNewActivation(t *testing.T) {
	t.Run("valid_activations", func(t *testing.T) {
		t.Parallel()
		tests := []struct {
			name string
			want string
		}{
			{"relu", "ReLU"},
			{"sigmoid", "Sigmoid"},
			{"linear", "Linear"},
			{"tanh", "Tanh"},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got, err := NewActivation(tc.name)
				if err != nil {
					t.Fatalf("NewActivation(%q) returned unexpected error: %v", tc.name, err)
				}

				if got.String() != tc.want {
					t.Errorf("NewActivation(%q).String() = %q; want %q", tc.name, got.String(), tc.want)
				}
			})
		}
	})

	t.Run("invalid_activations", func(t *testing.T) {
		t.Parallel()

		tests := []string{"unknown", "", "ReLu", "Sigmoid", " relu "}

		for _, tc := range tests {
			got, err := NewActivation(tc)
			if err == nil {
				t.Fatalf("NewActivation(%q) should return error", tc)
			}
			if got != nil {
				t.Errorf("NewActivation(%q) should return nil activation with error", tc)
			}
		}
	})
}

// ============================================================================
// UNIT TESTS - ReLU
// ============================================================================

func TestReLU(t *testing.T) {
	relu := ReLU{}

	t.Run("forward", func(t *testing.T) {
		tests := []struct {
			name    string
			input   float64
			want    float64
			wantNaN bool
		}{
			{"negative", -1, 0, false},
			{"zero", 0, 0, false},
			{"positive", 2, 2, false},
			{"positive_infinity", math.Inf(1), math.Inf(1), false},
			{"negative_infinity", math.Inf(-1), 0, false},
			{"nan", math.NaN(), math.NaN(), true},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := relu.Forward(tc.input)
				if tc.wantNaN {
					if !math.IsNaN(got) {
						t.Errorf("relu(%v) = %v; want NaN", tc.input, got)
					}
				} else if got != tc.want {
					t.Errorf("relu(%v) = %v; want %v", tc.input, got, tc.want)
				}
			})
		}
	})

	t.Run("string", func(t *testing.T) {
		t.Parallel()
		if got := relu.String(); got != "ReLU" {
			t.Errorf("ReLU.String() = %s; want ReLU", got)
		}
	})
}

// ============================================================================
// UNIT TESTS - Sigmoid
// ============================================================================

func TestSigmoid(t *testing.T) {
	sigmoid := Sigmoid{}

	t.Run("forward", func(t *testing.T) {
		tests := []struct {
			name    string
			input   float64
			want    float64
			wantNaN bool
		}{
			{"zero", 0, 0.5, false},
			{"small positive", 1, 0.7310586, false},
			{"small negative", -1, 0.2689414, false},
			{"large positive", 10, 0.9999546, false},
			{"large negative", -10, 0.0000454, false},
			{"positive_infinity", math.Inf(1), 1, false},
			{"negative_infinity", math.Inf(-1), 0, false},
			{"nan", math.NaN(), math.NaN(), true},
		}
		epsilon := 0.00001
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := sigmoid.Forward(tc.input)
				if tc.wantNaN {
					if !math.IsNaN(got) {
						t.Errorf("Sigmoid.Forward(%v) = %v; want NaN", tc.input, got)
					}
				}
				assertFloat64Equal(t, got, tc.want, epsilon, tc.name)
			})
		}
	})

	t.Run("string", func(t *testing.T) {
		t.Parallel()
		if got := sigmoid.String(); got != "Sigmoid" {
			t.Errorf("Sigmoid.String() = %s; want Sigmoid", got)
		}
	})
}

// ============================================================================
// UNIT TESTS - Linear
// ============================================================================

func TestLinear(t *testing.T) {
	linear := Linear{}

	t.Run("forward", func(t *testing.T) {
		tests := []struct {
			name    string
			input   float64
			want    float64
			wantNaN bool
		}{
			{"positive", 1, 1, false},
			{"negative", -1, -1, false},
			{"large_positive", 20, 20, false},
			{"large_negative", -100, -100, false},
			{"zero", 0, 0, false},
			{"positive_infinity", math.Inf(1), math.Inf(1), false},
			{"negative_infinity", math.Inf(-1), math.Inf(-1), false},
			{"nan", math.NaN(), math.NaN(), true},
		}

		epsilon := 0.0
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := linear.Forward(tc.input)
				if tc.wantNaN {
					if !math.IsNaN(got) {
						t.Errorf("Linear.Forward(%v) = %v; want NaN", tc.input, got)
					}
				}
				assertFloat64Equal(t, got, tc.want, epsilon, tc.name)
			})
		}
	})

	t.Run("string", func(t *testing.T) {
		t.Parallel()
		if got := linear.String(); got != "Linear" {
			t.Errorf("Linear.String() = %s; want Linear", got)
		}
	})
}

// ============================================================================
// UNIT TESTS - Tanh
// ============================================================================

func TestTanh(t *testing.T) {
	tanh := Tanh{}

	t.Run("forward", func(t *testing.T) {
		tests := []struct {
			name    string
			input   float64
			want    float64
			wantNaN bool
		}{
			{"zero", 0, 0, false},
			{"negative_one", -1, -0.7616, false},
			{"positive_one", 1, 0.7616, false},
			{"large_positive", 10, 1, false},
			{"large_negative", -10, -1, false},
			{"very_large_positive", 100, 1, false},
			{"very_large_negative", -100, -1, false},
			{"positive_infinity", math.Inf(1), 1, false},
			{"negative_infinity", math.Inf(-1), -1, false},
			{"nan", math.NaN(), math.NaN(), true},
		}

		epsilon := 0.0001

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := tanh.Forward(tc.input)
				if tc.wantNaN {
					if !math.IsNaN(got) {
						t.Errorf("Tanh.Forward(%v) = %v; want NaN", tc.input, got)
					}
				}
				assertFloat64Equal(t, got, tc.want, epsilon, tc.name)
			})
		}
	})

	t.Run("string", func(t *testing.T) {
		t.Parallel()
		if got := tanh.String(); got != "Tanh" {
			t.Errorf("Tanh.String() = %s; want Tanh", got)
		}
	})
}

// ============================================================================
// PROPERTY TESTS
// ============================================================================

func TestActivationProperties(t *testing.T) {
	t.Run("relu_properties", func(t *testing.T) {
		t.Parallel()
		relu := ReLU{}

		t.Run("non_negative", func(t *testing.T) {
			t.Parallel()
			// Property: ReLU output is always >= 0
			inputs := []float64{-1000, -100, -10, -1, -0.1, 0, 0.1, 1, 10, 100, 1000}

			for _, x := range inputs {
				if y := relu.Forward(x); y < 0 && !math.IsNaN(y) {
					t.Errorf("ReLU(%v) = %v; should be non-negative", x, y)
				}
			}
		})

		t.Run("identity_for_positive", func(t *testing.T) {
			t.Parallel()
			// Property: For x >= 0, relu(x) = x
			inputs := []float64{0, 0.1, 1, 5, 100, 1000}
			for _, x := range inputs {
				if y := relu.Forward(x); y != x {
					t.Errorf("ReLU(%v) = %v; should equal input", x, y)
				}
			}
		})

		t.Run("zero_for_negative", func(t *testing.T) {
			t.Parallel()
			// Property: For x < 0, relu(x) = 0
			inputs := []float64{-0.1, -1, -5, -100, -1000}
			for _, x := range inputs {
				if y := relu.Forward(x); y != 0 {
					t.Errorf("ReLU(%v) = %v; should be 0", x, y)
				}
			}
		})
	})

	t.Run("sigmoid_properties", func(t *testing.T) {
		t.Parallel()
		sigmoid := Sigmoid{}

		t.Run("output_range", func(t *testing.T) {
			// Property: Sigmoid outputs are always in (0, 1) for finite inputs
			inputs := []float64{math.Inf(-1), -1000, -100, -10, -1, 0, 1, 10, 100, 1000, math.Inf(1)}

			for _, x := range inputs {
				if y := sigmoid.Forward(x); y < 0 || y > 1 {
					t.Errorf("Sigmoid(%v) = %v; should be in [0,1]", x, y)
				}
			}
		})

		t.Run("symmetry", func(t *testing.T) {
			// Property: sigmoid(x) + sigmoid(-x) = 1
			inputs := []float64{0.5, 1, 2, 5, 10}

			for _, x := range inputs {
				pos := sigmoid.Forward(x)
				neg := sigmoid.Forward(-x)
				assertFloat64Equal(t, pos+neg, 1, 1e-10, "Sigmoid Symmetry")
			}
		})

		t.Run("midpoint", func(t *testing.T) {
			t.Parallel()
			// Property: sigmoid(0) = 0.5
			y := sigmoid.Forward(0)

			assertFloat64Equal(t, y, 0.5, 0.0, "Sigmoid Midpoint")
		})
	})

	t.Run("linear_properties", func(t *testing.T) {
		t.Parallel()
		linear := Linear{}
		t.Run("identity", func(t *testing.T) {
			t.Parallel()
			// Property: Linear activation is the identity function
			inputs := []float64{-1000, -100, -1, 0, 1, 100, 1000, math.Pi, -math.E}

			for _, x := range inputs {
				if y := linear.Forward(x); y != x {
					t.Errorf("Linear(%v) = %v; should be equal", x, y)
				}
			}
		})
		t.Run("additivity", func(t *testing.T) {
			t.Parallel()
			// Property: linear(x + y) = linear(x) + linear(y)
			inputPairs := []struct{ x, y float64 }{
				{1, 2}, {-1, 3}, {0, 5}, {-10, -5},
			}

			for _, pair := range inputPairs {
				sum := linear.Forward(pair.x + pair.y)
				seperate := linear.Forward(pair.x) + linear.Forward(pair.y)
				if sum != seperate {
					t.Errorf("Linear(%v + %v)", pair.x, pair.y)
				}
			}
		})

		t.Run("homogeneity", func(t *testing.T) {
			t.Parallel()
			// Property: linear(c * x) = c * linear(x)
			inputPairs := []struct{ c, x float64 }{
				{2, 3}, {-1, 5}, {0, 10}, {0.5, 4},
			}

			for _, pair := range inputPairs {
				scaled := linear.Forward(pair.c * pair.x)
				separate := pair.c * linear.Forward(pair.x)

				if scaled != separate {
					t.Errorf("Linear(%v * %v) = %v", pair.c, pair.x, scaled)
				}
			}
		})
	})

	t.Run("tanh_properties", func(t *testing.T) {
		t.Parallel()
		tanh := Tanh{}

		t.Run("output_range", func(t *testing.T) {
			// Property: Tanh outputs are always in (-1, 1) for finite inputs
			inputs := []float64{-1000, -100, -10, -1, 0, 1, 10, 100, 1000}
			for _, x := range inputs {
				if y := tanh.Forward(x); y < -1 || y > 1 {
					t.Errorf("Tanh(%v) = %v; should be in [-1,1]", x, y)
				}
			}
		})
		t.Run("odd_function", func(t *testing.T) {
			// Property: tanh(-x) = -tanh(x) (odd function)
			inputs := []float64{0.5, 1, 2, 5, 10}
			for _, x := range inputs {
				pos := tanh.Forward(x)
				neg := tanh.Forward(-x)
				assertFloat64Equal(t, pos, -neg, 1e-10, "Tanh")
			}
		})
		t.Run("zero_at_origin", func(t *testing.T) {
			// Property: tanh(0) = 0
			if y := tanh.Forward(0); math.Abs(y) > 1e-10 {
				t.Errorf("Tanh(0) = %v; want 0", y)
			}
		})
		t.Run("extreme_values", func(t *testing.T) {
			// Property: tanh approaches -1 and 1 at extremes
			largeNeg := tanh.Forward(-100)
			largePos := tanh.Forward(100)

			if math.Abs(largeNeg-(-1)) > 1e-10 {
				t.Errorf("tanh(-100) = %v; want very close to -1", largeNeg)
			}

			if math.Abs(largePos-1) > 1e-10 {
				t.Errorf("tanh(100) = %v; want very close to 1", largePos)
			}
		})
	})

	t.Run("all_activations_monotonic", func(t *testing.T) {
		t.Parallel()
		activations := []ActivationFunc{ReLU{}, Sigmoid{}, Tanh{}, Linear{}}

		for _, act := range activations {
			t.Run(act.String(), func(t *testing.T) {
				// All our activations are monotonically increasing
				x1, x2 := -5.0, 5.0
				y1, y2 := act.Forward(x1), act.Forward(x2)
				if y1 > y2 {
					t.Errorf("%s is not monotonic: f(%v)=%v > f(%v)=%v",
						act.String(), x1, y1, x2, y2)
				}
			})
		}
	})
}

// ============================================================================
// ROBUSTNESS TESTS
// ============================================================================
func TestActivationRobustness(t *testing.T) {
	t.Run("extreme_values", func(t *testing.T) {
		extremeInputs := []struct {
			name  string
			value float64
		}{
			{"very_large_positive", 1e100},
			{"very_large_negative", -1e100},
			{"very_small_positive", 1e-100},
			{"very_small_negative", -1e-100},
			{"max_float64", math.MaxFloat64},
			{"smallest_normal_float64", math.SmallestNonzeroFloat64},
			{"positive_infinity", math.Inf(1)},
			{"negative_infinity", math.Inf(-1)},
			{"nan", math.NaN()},
			{"negative_zero", math.Copysign(0, -1)},
		}

		for _, actName := range getActivationNames() {
			t.Run(actName, func(t *testing.T) {
				t.Parallel()
				for _, input := range extremeInputs {
					t.Run(input.name, func(t *testing.T) {
						func() {
							defer func() {
								if r := recover(); r != nil {
									t.Errorf("%s panicked with input %s (%v): %v",
										actName, input.name, input.value, r)
								}
							}()

							act, _ := NewActivation(actName)
							result := act.Forward(input.value)

							t.Logf("%s(%s) = %v", actName, input.name, result)
						}()
					})
				}
			})
		}
	})
	t.Run("bounds_with_extreme_inputs", func(t *testing.T) {
		t.Parallel()

		extremeInputs := []float64{1000, -1000, math.MaxFloat64, -math.MaxFloat64}

		t.Run("sigmoid_bounds", func(t *testing.T) {
			t.Parallel()
			sigmoid, _ := NewActivation("sigmoid")

			for _, input := range extremeInputs {
				result := sigmoid.Forward(input)
				// Should stay in [0,1] even with extreme inputs
				if !math.IsNaN(result) && (result < 0 || result > 1) {
					t.Errorf("Sigmoid(%v) out of bounds [0,1]: %v", input, result)
				}
			}
		})

		t.Run("tanh_bounds", func(t *testing.T) {
			t.Parallel()
			tanh, _ := NewActivation("tanh")

			for _, input := range extremeInputs {
				result := tanh.Forward(input)
				// Should stay in [-1,1] even with extreme inputs
				if !math.IsNaN(result) && (result < -1 || result > 1) {
					t.Errorf("tanh(%v) out of bounds [-1,1]: %v", input, result)
				}
			}
		})

		t.Run("relu_non_negative", func(t *testing.T) {
			t.Parallel()
			relu, _ := NewActivation("relu")

			for _, input := range extremeInputs {
				result := relu.Forward(input)
				// Should be positive even with extreme inputs
				if !math.IsNaN(result) && result < 0 {
					t.Errorf("ReLU(%v) should be non-negative: %v", input, result)
				}
			}
		})
	})

	t.Run("nan_propagation", func(t *testing.T) {
		t.Parallel()

		for _, actName := range getActivationNames() {
			t.Run(actName, func(t *testing.T) {
				t.Parallel()
				act, _ := NewActivation(actName)

				// NaN should propagate through all activation functions
				result := act.Forward(math.NaN())
				if !math.IsNaN(result) {
					t.Errorf("%s should propagate NaN, got %v", actName, result)
				}
			})
		}
	})
}

// ============================================================================
// BENCHMARKS
// ============================================================================

func BenchmarkActivationForward(b *testing.B) {
	inputs := []float64{-5.0, -1.0, 0.0, 1.0, 5.0}

	for _, actName := range getActivationNames() {
		act, _ := NewActivation(actName)

		b.Run(actName, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, input := range inputs {
					_ = act.Forward(input)
				}
			}
		})
	}
}

func BenchmarkActivationMemory(b *testing.B) {
	for _, actName := range getActivationNames() {
		act, _ := NewActivation(actName)

		b.Run(actName, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = act.Forward(1.5)
			}
		})
	}
}

func BenchmarkNewActivation(b *testing.B) {
	for _, actName := range getActivationNames() {
		b.Run(actName, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = NewActivation(actName)
			}
		})
	}
}

func BenchmarkNewActivationMemory(b *testing.B) {
	for _, actName := range getActivationNames() {
		b.Run(actName, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = NewActivation(actName)
			}
		})
	}
}

func BenchmarkBatchProcessing(b *testing.B) {
	batchSizes := []struct {
		name string
		size int
	}{
		{"small_batch", 100},
		{"medium_batch", 1000},
		{"large_batch", 10000},
	}

	for _, batch := range batchSizes {
		inputs := make([]float64, batch.size)

		for i := range inputs {
			inputs[i] = float64(i) * 0.01
		}

		b.Run(batch.name, func(b *testing.B) {
			for _, actName := range getActivationNames() {
				b.Run(actName, func(b *testing.B) {
					b.ResetTimer()
					act, _ := NewActivation(actName)
					for i := 0; i < b.N; i++ {
						for j := 0; j < batch.size; j++ {
							act.Forward(inputs[j])
						}
					}
				})
			}
		})
	}
}
