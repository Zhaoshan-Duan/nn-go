package activation

import (
	"math"
	"testing"
)

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
				t.Parallel()
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
		want := "ReLU"
		got := relu.String()
		if got != want {
			t.Errorf("ReLU.String() = %s; want %s", got, want)
		}
	})

	t.Run("properties", func(t *testing.T) {
		t.Parallel()

		t.Run("non_negative_output", func(t *testing.T) {
			t.Parallel()
			// Property: ReLU output is always >= 0
			testInputs := []float64{-1000, -100, -10, -1, -0.1, 0, 0.1, 1, 10, 100, 1000}

			for _, input := range testInputs {
				output := relu.Forward(input)
				if !math.IsNaN(output) && output < 0 {
					t.Errorf("ReLU output should be non-negative : relu(%v) = %v", input, output)
				}
			}
		})

		t.Run("monotonic", func(t *testing.T) {
			t.Parallel()
			// Property: If x1 <= x2, then relu(x1) <= relu(x2)
			testPairs := []struct{ x1, x2 float64 }{
				{-5, -2}, {-1, 0}, {0, 1}, {1, 5}, {-10, 10},
			}

			for _, pair := range testPairs {
				y1 := relu.Forward(pair.x1)
				y2 := relu.Forward(pair.x2)

				if y1 > y2 {
					t.Errorf("ReLU should be monotonic: relu(%v)=%v > relu(%v)=%v", pair.x1, y1, pair.x2, y2)
				}
			}
		})

		t.Run("identity_for_positive", func(t *testing.T) {
			t.Parallel()
			// Property: For x >= 0, relu(x) = x
			positiveInputs := []float64{0, 0.1, 1, 5, 100, 1000}
			for _, input := range positiveInputs {
				output := relu.Forward(input)
				if output != input {
					t.Errorf("ReLU should be identity for positive: relu(%v) = %v, want %v", input, output, input)
				}
			}
		})

		t.Run("zero_for_negative", func(t *testing.T) {
			t.Parallel()
			// Property: For x < 0, relu(x) = 0
			negativeInputs := []float64{-0.1, -1, -5, -100, -1000}
			for _, input := range negativeInputs {
				output := relu.Forward(input)
				if output != 0 {
					t.Errorf("ReLU should be zero for negative: relu(%v) = %v, want 0", input, output)
				}
			}
		})
	})
}

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
				t.Parallel()
				got := sigmoid.Forward(tc.input)
				if tc.wantNaN {
					if !math.IsNaN(got) {
						t.Errorf("Sigmoid.Forward(%v) = %v; want NaN", tc.input, got)
					}
				} else if math.Abs(got-tc.want) > epsilon {
					t.Errorf("Sigmoid.Forward(%v) = %v, got %v", tc.input, got, tc.want)
				}
			})
		}
	})

	t.Run("string", func(t *testing.T) {
		t.Parallel()
		want := "Sigmoid"
		got := sigmoid.String()
		if got != want {
			t.Errorf("Sigmoid.String() = %s; want %s", got, want)
		}
	})

	t.Run("properties", func(t *testing.T) {
		t.Parallel()

		t.Run("output_range", func(t *testing.T) {
			t.Parallel()
			// Property: Sigmoid outputs are always in (0, 1) for finite inputs
			testInputs := []float64{math.Inf(-1), -1000, -100, -10, -1, 0, 1, 10, 100, 1000, math.Inf(1)}

			for _, input := range testInputs {
				output := sigmoid.Forward(input)
				if output < 0 || output > 1 {
					t.Errorf("Sigmoid output should be in (0,1): sigmoid(%v) = %v", input, output)
				}
			}
		})

		t.Run("monotonic", func(t *testing.T) {
			t.Parallel()
			// Property: Sigmoid is monotonically increasing
			testPairs := []struct{ x1, x2 float64 }{
				{-10, -5}, {-1, 0}, {0, 1}, {1, 5}, {-100, 100},
			}

			for _, pair := range testPairs {
				y1 := sigmoid.Forward(pair.x1)
				y2 := sigmoid.Forward(pair.x2)

				if y1 >= y2 {
					t.Errorf("Sigmoid should be monotonic: sigmoid(%v)=%v >= sigmoid(%v)=%v", pair.x1, y1, pair.x2, y2)
				}
			}
		})

		t.Run("symmetry", func(t *testing.T) {
			t.Parallel()
			// Property: sigmoid(x) + sigmoid(-x) = 1
			testInputs := []float64{0.5, 1, 2, 5, 10}

			for _, x := range testInputs {
				pos := sigmoid.Forward(x)
				neg := sigmoid.Forward(-x)
				sum := pos + neg

				if math.Abs(sum-1.0) > 1e-10 {
					t.Errorf("Sigmoid symmetry failed: sigmoid(%v) + sigmoid(%v) = %v; want 1.0", x, -x, sum)
				}
			}
		})

		t.Run("midpoint", func(t *testing.T) {
			t.Parallel()
			// Property: sigmoid(0) = 0.5
			output := sigmoid.Forward(0)
			want := 0.5

			if math.Abs(output-want) > 1e-10 {
				t.Errorf("sigmoid(0) = %v; want %v", output, want)
			}
		})
	})
}

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

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				got := linear.Forward(tc.input)
				if tc.wantNaN {
					if !math.IsNaN(got) {
						t.Errorf("Linear.Forward(%v) = %v; want NaN", tc.input, got)
					}
				} else if got != tc.want {
					t.Errorf("Linear.Forward(%v) = %v; want %v", tc.input, got, tc.want)
				}
			})
		}
	})

	t.Run("string", func(t *testing.T) {
		t.Parallel()
		want := "Linear"
		got := linear.String()
		if got != want {
			t.Errorf("Linear.String() = %s; want %s", got, want)
		}
	})

	t.Run("properties", func(t *testing.T) {
		t.Parallel()

		t.Run("identity_function", func(t *testing.T) {
			t.Parallel()
			// Property: Linear activation is the identity function
			testInputs := []float64{-1000, -100, -1, 0, 1, 100, 1000, math.Pi, -math.E}

			for _, input := range testInputs {
				output := linear.Forward(input)
				if output != input {
					t.Errorf("Linear should be identity: linear(%v) = %v; want %v", input, output, input)
				}
			}
		})

		t.Run("additivity", func(t *testing.T) {
			t.Parallel()
			// Property: linear(x + y) = linear(x) + linear(y)
			testPairs := []struct{ x, y float64 }{
				{1, 2}, {-1, 3}, {0, 5}, {-10, -5},
			}

			for _, pair := range testPairs {
				sum := linear.Forward(pair.x + pair.y)
				separate := linear.Forward(pair.x) + linear.Forward(pair.y)

				if sum != separate {
					t.Errorf("Linear additivity failed: linear(%v+%v) = %v; linear(%v)+linear(%v) = %v", pair.x, pair.y, sum, pair.x, pair.y, separate)
				}
			}
		})

		t.Run("homogeneity", func(t *testing.T) {
			t.Parallel()
			// Property: linear(c * x) = c * linear(x)
			testCases := []struct{ c, x float64 }{
				{2, 3}, {-1, 5}, {0, 10}, {0.5, 4},
			}

			for _, tc := range testCases {
				scaled := linear.Forward(tc.c * tc.x)
				separate := tc.c * linear.Forward(tc.x)

				if scaled != separate {
					t.Errorf("Linear homogeneity failed: linear(%v*%v) = %v; %v*linear(%v) = %v", tc.c, tc.x, scaled, tc.c, tc.x, separate)
				}
			}
		})
	})
}

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
				t.Parallel()
				got := tanh.Forward(tc.input)
				if tc.wantNaN {
					if !math.IsNaN(got) {
						t.Errorf("Tanh.Forward(%v) = %v; want NaN", tc.input, got)
					}
				} else if math.Abs(got-tc.want) > epsilon {
					t.Errorf("Tanh.Forward(%v) = %v; want %v", tc.input, got, tc.want)
				}
			})
		}
	})

	t.Run("string", func(t *testing.T) {
		t.Parallel()
		want := "Tanh"
		got := tanh.String()
		if got != want {
			t.Errorf("Tanh.String() = %s; want %s", got, want)
		}
	})

	t.Run("properties", func(t *testing.T) {
		t.Parallel()

		t.Run("output_range", func(t *testing.T) {
			t.Parallel()
			// Property: Tanh outputs are always in (-1, 1) for finite inputs
			testInputs := []float64{-1000, -100, -10, -1, 0, 1, 10, 100, 1000}

			for _, input := range testInputs {
				output := tanh.Forward(input)
				if output < -1 || output > 1 {
					t.Errorf("Tanh output should be in [-1,1]: tanh(%v) = %v", input, output)
				}
			}
		})

		t.Run("monotonic", func(t *testing.T) {
			t.Parallel()
			// Property: Tanh is monotonically increasing
			testPairs := []struct{ x1, x2 float64 }{
				{-10, -5}, {-1, 0}, {0, 1}, {1, 5}, {-100, 100},
			}

			for _, pair := range testPairs {
				y1 := tanh.Forward(pair.x1)
				y2 := tanh.Forward(pair.x2)

				if y1 >= y2 {
					t.Errorf("Tanh should be monotonic: tanh(%v)=%v >= tanh(%v)=%v",
						pair.x1, y1, pair.x2, y2)
				}
			}
		})

		t.Run("odd_function", func(t *testing.T) {
			t.Parallel()
			// Property: tanh(-x) = -tanh(x) (odd function)
			testInputs := []float64{0.5, 1, 2, 5, 10}

			for _, x := range testInputs {
				pos := tanh.Forward(x)
				neg := tanh.Forward(-x)

				if math.Abs(pos+neg) > 1e-10 {
					t.Errorf("Tanh odd function property failed: tanh(%v) = %v, tanh(%v) = %v",
						x, pos, -x, neg)
				}
			}
		})

		t.Run("zero_at_origin", func(t *testing.T) {
			t.Parallel()
			// Property: tanh(0) = 0
			output := tanh.Forward(0)
			if math.Abs(output) > 1e-10 {
				t.Errorf("tanh(0) = %v; want 0", output)
			}
		})

		t.Run("extreme_values", func(t *testing.T) {
			t.Parallel()
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
}

func TestNewActivation(t *testing.T) {
	t.Run("valid_activations", func(t *testing.T) {
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
				t.Parallel()
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
		tests := []struct {
			name  string
			input string
		}{
			{"unknown", "unknown"},
			{"empty", ""},
			{"typo_relu", "ReLu"},
			{"typo_sigmoid", "Sigmoid"},
			{"with_spaces", " relu "},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				got, err := NewActivation(tc.input)
				if err == nil {
					t.Fatalf("NewActivation(%q) = %v, nil; want nil, error", tc.input, got)
				}
				if got != nil {
					t.Errorf("NewActivation(%q) returned non-nil activation with error", tc.input)
				}
			})
		}
	})
}

func BenchmarkActivationInputRanges(b *testing.B) {
	inputs := []struct {
		name  string
		value float64
	}{
		{"small_positive", 0.1},
		{"medium_positive", 5.0},
		{"large_positive", 100.0},
		{"small_negative", -0.1},
		{"medium_negative", -5.0},
		{"large_negative", -100.0},
	}

	b.Run("ReLU", func(b *testing.B) {
		relu := ReLU{}
		for _, input := range inputs {
			b.Run(input.name, func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					relu.Forward(input.value)
				}
			})
		}
	})

	b.Run("Sigmoid", func(b *testing.B) {
		sigmoid := Sigmoid{}
		for _, input := range inputs {
			b.Run(input.name, func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					sigmoid.Forward(input.value)
				}
			})
		}
	})

	b.Run("Linear", func(b *testing.B) {
		linear := Linear{}
		for _, input := range inputs {
			b.Run(input.name, func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					linear.Forward(input.value)
				}
			})
		}
	})

	b.Run("Tanh", func(b *testing.B) {
		tanh := Tanh{}
		for _, input := range inputs {
			b.Run(input.name, func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					tanh.Forward(input.value)
				}
			})
		}
	})
}

func BenchmarkActivationMemoryAllocation(b *testing.B) {
	scenarios := []struct {
		name  string
		value float64
		why   string
	}{
		{"normal", 2.5, "typical usage"},
		{"large", 1000.0, "might trigger overflow handling"},
		{"infinity", math.Inf(1), "special value handling"},
		{"nan", math.NaN(), "NaN propagation"},
		{"tiny", 1e-100, "underflow scenarios"},
	}

	b.Run("ReLU", func(b *testing.B) {
		relu := ReLU{}

		for _, scenario := range scenarios {
			b.Run(scenario.name, func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = relu.Forward(scenario.value)
				}
			})
		}
	})

	b.Run("Sigmoid", func(b *testing.B) {
		sigmoid := Sigmoid{}
		for _, scenario := range scenarios {
			b.Run(scenario.name, func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = sigmoid.Forward(scenario.value)
				}
			})
		}
	})

	b.Run("Linear", func(b *testing.B) {
		linear := Linear{}
		for _, scenario := range scenarios {
			b.Run(scenario.name, func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = linear.Forward(scenario.value)
				}
			})
		}
	})

	b.Run("Tanh", func(b *testing.B) {
		tanh := Tanh{}

		for _, scenario := range scenarios {
			b.Run(scenario.name, func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = tanh.Forward(scenario.value)
				}
			})
		}
	})
}

func BenchmarkNewActivation(b *testing.B) {
	activationNames := []string{"relu", "sigmoid", "linear", "tanh"}

	for _, name := range activationNames {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = NewActivation(name)
			}
		})
	}
}

func BenchmarkNewActivationMemory(b *testing.B) {
	activationNames := []string{"relu", "sigmoid", "linear", "tanh"}

	for _, name := range activationNames {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = NewActivation(name)
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
			b.Run("ReLU", func(b *testing.B) {
				relu := ReLU{}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := 0; j < batch.size; j++ {
						relu.Forward(inputs[j])
					}
				}
			})

			b.Run("Sigmoid", func(b *testing.B) {
				sigmoid := Sigmoid{}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := 0; j < batch.size; j++ {
						sigmoid.Forward(inputs[j])
					}
				}
			})

			b.Run("Linear", func(b *testing.B) {
				linear := Linear{}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := 0; j < batch.size; j++ {
						linear.Forward(inputs[j])
					}
				}
			})

			b.Run("Tanh", func(b *testing.B) {
				tanh := Tanh{}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := 0; j < batch.size; j++ {
						tanh.Forward(inputs[j])
					}
				}
			})
		})
	}
}
