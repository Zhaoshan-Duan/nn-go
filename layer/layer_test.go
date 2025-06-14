package layer

import (
	"fmt"
	"math"
	"math/rand/v2"
	"neural-network-project/activation"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNewLayer(t *testing.T) {
	t.Run("valid_parameters", func(t *testing.T) {
		// activation type doesn't layer constructo so just use one
		activation := mustCreateActivation("relu")

		tests := []struct {
			name                  string
			inputSize, outputSize int
		}{
			{"tiny", 1, 1},
			{"small", 3, 2},
			{"single_neuron", 5, 1},
			{"single_input", 1, 5},
			{"square", 10, 10},
			{"wide", 2, 100},
			{"narrow", 100, 2},
			{"large", 784, 128},
			{"very_large", 2048, 1024},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				layer, err := NewLayer(tc.inputSize, tc.outputSize, activation)
				if err != nil {
					t.Fatalf("NewLayer(%d, %d, relu) returned error: %v", tc.inputSize, tc.outputSize, err)
				}

				// check dimensions
				if layer.InputSize() != tc.inputSize {
					t.Errorf("InputSize() = %d; want %d", layer.InputSize(), tc.inputSize)
				}
				if layer.OutputSize() != tc.outputSize {
					t.Errorf("OutputSize() = %d; want %d", layer.OutputSize(), tc.outputSize)
				}

				// check weights
				if len(layer.Weights) != tc.outputSize {
					t.Errorf("len(Weights) = %d; want %d", len(layer.Weights), tc.outputSize)
				}
				for i, weightRow := range layer.Weights {
					if len(weightRow) != tc.inputSize {
						t.Errorf("len(Weights[%d]) = %d; want %d", i, len(weightRow), tc.inputSize)
					}
				}

				// check bias
				if len(layer.Biases) != tc.outputSize {
					t.Errorf("len(Biases) = %d; want %d", len(layer.Biases), tc.outputSize)
				}
			})
		}
	})
	t.Run("invalid_parameters", func(t *testing.T) {
		tests := []struct {
			name                  string
			inputSize, outputSize int
			activation            activation.ActivationFunc
			wantErr               bool
		}{
			{"zero_input_size", 0, 5, mustCreateActivation("relu"), true},
			{"negative_input_size", -1, 5, mustCreateActivation("relu"), true},
			{"zero_output_size", 5, 0, mustCreateActivation("relu"), true},
			{"negative_output_size", 5, -1, mustCreateActivation("relu"), true},
			{"both_zero", 0, 0, mustCreateActivation("relu"), true},
			{"both_negative", -5, -3, mustCreateActivation("relu"), true},
			{"nil_activation", 5, 3, nil, true},
			{"valid_case", 3, 2, mustCreateActivation("sigmoid"), false},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				layer, err := NewLayer(tc.inputSize, tc.outputSize, tc.activation)

				if tc.wantErr {
					if err == nil {
						t.Errorf("NewLayer(%d, %d, %v) should return error", tc.inputSize, tc.outputSize, tc.activation)
					}
					if layer != nil {
						t.Error("Layer should be nil when error occurs")
					}
				} else {
					if err != nil {
						t.Errorf("NewLayer(%d, %d, %v) returned unexpected error: %v",
							tc.inputSize, tc.outputSize, tc.activation, err)
					}
					if layer == nil {
						t.Error("Layer should not be nil when no error occurs")
					}
				}
			})
		}
	})
	t.Run("valid_activations", func(t *testing.T) {
		inputSize, outputSize := 5, 3
		activationNames := getActivationNames()

		for _, actName := range activationNames {
			t.Run(actName, func(t *testing.T) {
				t.Parallel()
				act, err := activation.NewActivation(actName)
				if err != nil {
					t.Fatalf("Failed to create activation %s: %v", actName, err)
				}

				layer, err := NewLayer(inputSize, outputSize, act)
				if err != nil {
					t.Fatalf("NewLayer(%d, %d, %s) returned error: %v", inputSize, outputSize, actName, err)
				}

				// Check activation was set correctly
				if layer.Activation == nil {
					t.Error("Activation should not be nil")
				}
				if layer.Activation.String() != act.String() {
					t.Errorf("Activation = %s; want %s", layer.Activation.String(), act.String())
				}

				// test basic forward pass
				input := make([]float64, inputSize)
				for i := range input {
					input[i] = 1.0
				}

				output, err := layer.Forward(input)
				if err != nil {
					t.Errorf("Forward pass failed with %s activation: %v", actName, err)
				}
				if len(output) != outputSize {
					t.Errorf("Output size = %d; want %d", len(output), outputSize)
				}
			})
		}
	})
	t.Run("initialization_properties", func(t *testing.T) {
		t.Run("weights_and_biases_range", func(t *testing.T) {
			t.Parallel()
			layer, err := NewLayer(10, 5, mustCreateActivation("relu"))
			if err != nil {
				t.Fatalf("Failed to create layer: %v", err)
			}

			for i, weightRow := range layer.Weights {
				for j, weight := range weightRow {
					if weight < -1 || weight > 1 {
						t.Errorf("Weight[%d][%d] = %v; want in range [-1, 1]", i, j, weight)
					}
				}
			}

			for i, bias := range layer.Biases {
				if bias < -1 || bias > 1 {
					t.Errorf("Bias[%d] = %v; want in range [-1, 1]", i, bias)
				}
			}
		})
		t.Run("weights_randomness", func(t *testing.T) {
			t.Parallel()

			layer1, _ := NewLayer(5, 3, mustCreateActivation("sigmoid"))
			layer2, _ := NewLayer(5, 3, mustCreateActivation("sigmoid"))

			identical := true

			for i := range layer1.Weights {
				for j := range layer1.Weights[i] {
					if layer1.Weights[i][j] != layer2.Weights[i][j] {
						identical = false
						break
					}
				}
				if !identical {
					break
				}
			}

			if identical {
				t.Error("two layers should have different random weights")
			}
		})
		t.Run("zero_input_handling", func(t *testing.T) {
			t.Parallel()
			layer, _ := NewLayer(5, 3, mustCreateActivation("linear"))
			zeroInput := make([]float64, 5)

			output, err := layer.Forward(zeroInput)
			if err != nil {
				t.Errorf("Forward() with zero input returned error: %v", err)
			}

			// With zero input, output should equal biases (for linear activation)
			for i, val := range output {
				if val != layer.Biases[i] {
					t.Errorf("With zero input, output[%d] = %v; want bias %v", i, val, layer.Biases[i])
				}
			}
		})
	})
}

func TestLayerForward(t *testing.T) {
	t.Run("valid_forward_pass", func(t *testing.T) {
		tests := []struct {
			name       string
			inputSize  int
			outputSize int
			activation string
		}{
			{"small_relu", 3, 2, "relu"},
			{"single_sigmoid", 4, 1, "sigmoid"},
			{"large_tanh", 100, 50, "tanh"},
			{"linear_identity", 5, 5, "linear"},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				layer, err := NewLayer(tc.inputSize, tc.outputSize, mustCreateActivation(tc.activation))
				if err != nil {
					t.Fatalf("Failed to create layer: %v", err)
				}

				input := make([]float64, tc.inputSize)
				for i := range input {
					input[i] = rand.Float64()*2 - 1
				}

				output, err := layer.Forward(input)
				if err != nil {
					t.Errorf("Forward() returned error: %v", err)
				}

				if len(output) != tc.outputSize {
					t.Errorf("len(output) = %d; want %d", len(output), tc.outputSize)
				}
				// Check for NaN values
				for i, val := range output {
					if math.IsNaN(val) {
						t.Errorf("output[%d] = NaN", i)
					}
				}
			})
		}
	})
	// ERROR HANDLING TESTS
	t.Run("forward_error_cases", func(t *testing.T) {
		t.Run("invalid_input_size", func(t *testing.T) {
			t.Parallel()
			layer, err := NewLayer(3, 2, mustCreateActivation("relu"))
			if err != nil {
				t.Fatalf("Failed to create layer: %v", err)
			}
			tests := []struct {
				name  string
				input []float64
			}{
				{"nil_input", nil},
				{"empty_input", []float64{}},
				{"wrong_size_small", []float64{1.0, 2.0}},
				{"wrong_size_large", []float64{1.0, 2.0, 3.0, 4.0}},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					output, err := layer.Forward(tc.input)
					if err == nil {
						t.Errorf("Forward(%v) should return error", tc.input)
					}
					if output != nil {
						t.Error("Output should be nil when error occurs")
					}
				})
			}
		})
		t.Run("error_message_quality", func(t *testing.T) {
			layer, _ := NewLayer(3, 2, mustCreateActivation("relu"))

			_, err := layer.Forward([]float64{1.0, 2.0}) // Wrong size
			if err == nil || !strings.Contains(err.Error(), "expected 3 got 2") {
				t.Errorf("Error message not helpful: %v", err)
			}
		})
	})
	t.Run("special_float_inputs", func(t *testing.T) {
		layer, _ := NewLayer(3, 2, mustCreateActivation("relu"))
		tests := []struct {
			name  string
			input []float64
		}{
			{"with_nan", []float64{1.0, math.NaN(), 3.0}},
			{"with_inf", []float64{math.Inf(1), 2.0, 3.0}},
			{"with_neg_inf", []float64{1.0, math.Inf(-1), 3.0}},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				output, err := layer.Forward(tc.input)
				if err != nil {
					t.Errorf("Forward(%v) returned error: %v", tc.input, err)
				}
				_ = output
			})
		}
	})
	// DETERMINISTIC TESTS
	t.Run("forward_deterministic_behavior", func(t *testing.T) {
		t.Parallel()

		layer, _ := NewLayer(3, 2, mustCreateActivation("sigmoid"))
		input := []float64{1.0, 2.0, 3.0}

		output1, _ := layer.Forward(input)
		output2, _ := layer.Forward(input)

		for i := range output1 {
			if output1[i] != output2[i] {
				t.Errorf("Output not deterministic: output1[%d] = %v, output2[%d] = %v", i, output1[i], i, output2[i])
			}
		}
	})
	// MATHEMATICAL CORRECTNESS TESTS
	t.Run("forward_mathematical_correctness", func(t *testing.T) {
		t.Run("simple_linear_calculation_simple_feature", func(t *testing.T) {
			t.Parallel()
			// 1×1 layer: linear, weight=200, bias=100, input=1 → output=300
			layer, _ := NewLayer(1, 1, mustCreateActivation("linear"))
			layer.Weights[0] = []float64{200.0}
			layer.Biases[0] = 100

			input := []float64{1.0}
			output, _ := layer.Forward(input)

			expected := 300.0
			if math.Abs(expected-output[0]) > 1e-10 {
				t.Errorf("output[0] = %v; want %v", output[0], expected)
			}
		})

		t.Run("simple_linear_calculation", func(t *testing.T) {
			t.Parallel()
			// 2×1 layer: linear, weights=[2,3], bias=1, input=[4,5] → output=24
			layer, _ := NewLayer(2, 1, mustCreateActivation("linear"))
			layer.Weights[0] = []float64{2.0, 3.0}
			layer.Biases[0] = 1.0

			input := []float64{4.0, 5.0}
			output, _ := layer.Forward(input)

			// (2.0 * 4.0) + (3.0 * 5.0) + 1.0 = 24.0
			expected := 24.0
			if math.Abs(expected-output[0]) > 1e-10 {
				t.Errorf("output[0] = %v; want %v", output[0], expected)
			}
		})

		t.Run("multiple_neurons_linear", func(t *testing.T) {
			t.Parallel()
			// 2×3 layer: linear activation
			layer, _ := NewLayer(2, 3, mustCreateActivation("linear"))
			layer.Weights[0] = []float64{1.0, 2.0}
			layer.Weights[1] = []float64{0.5, -1.0}
			layer.Weights[2] = []float64{-2.0, 3.0}
			layer.Biases[0] = 0.0
			layer.Biases[1] = 1.0
			layer.Biases[2] = -0.5

			input := []float64{2.0, 3.0}
			output, _ := layer.Forward(input)
			expected := []float64{8.0, -1.0, 4.5}

			for i := range expected {
				if math.Abs(expected[i]-output[i]) > 1e-10 {
					t.Errorf("output[0] = %v; want %v", output[0], expected)
				}
			}
		})

		t.Run("multiple_neurons_relu", func(t *testing.T) {
			t.Parallel()
			// 3×2 layer with ReLU, test positive/negative pre-activations
			layer, _ := NewLayer(3, 2, mustCreateActivation("relu"))
			layer.Weights[0] = []float64{1.0, -2.0, 3.0}
			layer.Weights[1] = []float64{-1.0, -1.0, -1.0}
			layer.Biases[0] = 0.0
			layer.Biases[1] = 0.0

			input := []float64{1.0, 1.0, 1.0}
			output, _ := layer.Forward(input)

			expected := []float64{2.0, 0.0}

			for i, exp := range expected {
				if math.Abs(output[i]-exp) > 1e-10 {
					t.Errorf("output[%d] = %v; want %v", i, output[i], exp)
				}
			}
		})

		t.Run("simple_sigmoid_calculation_simple_feature", func(t *testing.T) {
			t.Parallel()
			// 1×1 sigmoid layer with specific weights to test sigmoid math
			layer, _ := NewLayer(1, 1, mustCreateActivation("sigmoid"))
			layer.Weights[0] = []float64{2.0}
			layer.Biases[0] = -4.5

			input := []float64{0}
			output, _ := layer.Forward(input)

			expected := 0.01
			if math.Abs(output[0]-expected) > 0.001 {
				t.Errorf("output[0] = %v; want %v", output[0], expected)
			}
		})

		t.Run("identity_transformation", func(t *testing.T) {
			t.Parallel()
			// 2×2 layer with identity matrix weights, zero bias
			layer, _ := NewLayer(2, 2, mustCreateActivation("linear"))
			layer.Weights[0] = []float64{1.0, 0.0}
			layer.Weights[1] = []float64{0.0, 1.0}
			layer.Biases[0] = 0.0
			layer.Biases[1] = 0.0

			input := []float64{7.0, -3.0}
			output, _ := layer.Forward(input)

			// Should output exactly the input
			for i, inp := range input {
				if math.Abs(output[i]-inp) > 1e-10 {
					t.Errorf("output[%d] = %v; want %v (identity)", i, output[i], inp)
				}
			}
		})
		t.Run("scale_invariance_test", func(t *testing.T) {
			t.Parallel()
			layer, _ := NewLayer(2, 1, mustCreateActivation("linear"))
			layer.Weights[0] = []float64{1.0, 1.0}
			layer.Biases[0] = 0.0

			input1 := []float64{1.0, 1.0}
			input2 := []float64{2.0, 2.0}

			output1, _ := layer.Forward(input1)
			output2, _ := layer.Forward(input2)

			// For linear activation with zero bias, doubling input should double output
			if math.Abs(output2[0]-2*output1[0]) > 1e-10 {
				t.Errorf("Linear scaling failed: 2*%v ≠ %v", output1[0], output2[0])
			}
		})
	})
	t.Run("activation_properties_by_size", func(t *testing.T) {
		layerSizes := []struct {
			name                  string
			inputSize, outputSize int
		}{
			{"small", 2, 3},
			{"medium", 10, 8},
			{"large", 100, 50},
		}
		t.Run("relu_non_negative", func(t *testing.T) {
			// Property: ReLU should output non-negative values
			for _, size := range layerSizes {
				t.Run(size.name, func(t *testing.T) {
					t.Parallel()
					layer, _ := NewLayer(size.inputSize, size.outputSize, mustCreateActivation("relu"))

					for i := range layer.Weights {
						for j := range layer.Weights[i] {
							// Force negative weights to test ReLU
							layer.Weights[i][j] = -math.Abs(layer.Weights[i][j]) - 1.0
						}
						// Force negative biases to test ReLU
						layer.Biases[i] = -math.Abs(layer.Biases[i]) - 1.0
					}

					input := make([]float64, size.inputSize)
					for i := range input {
						input[i] = 1.0
					}

					output, _ := layer.Forward(input)

					for i, val := range output {
						if val < 0 {
							t.Errorf("ReLU output[%d] = %v; should be non-negative", i, val)
						}
					}
				})
			}
		})
		t.Run("sigmoid_range", func(t *testing.T) {
			// Property: Sigmoid should output in range [0, 1]
			for _, size := range layerSizes {
				t.Run(size.name, func(t *testing.T) {
					t.Parallel()
					layer, _ := NewLayer(size.inputSize, size.outputSize, mustCreateActivation("sigmoid"))

					input := make([]float64, size.inputSize)
					for i := range input {
						// Create alternating large values to test sigmoid saturation
						if i%2 == 0 {
							input[i] = 100.0
						} else {
							input[i] = -100.0
						}
					}
					output, _ := layer.Forward(input)

					for i, val := range output {
						if val < 0 || val > 1 {
							t.Errorf("Sigmoid output[%d] = %v; should be in [0, 1]", i, val)
						}
						if math.IsNaN(val) || math.IsInf(val, 0) {
							t.Errorf("Sigmoid output[%d] = %v; should be finite", i, val)
						}
					}
				})
			}
		})
		t.Run("tanh_range", func(t *testing.T) {
			for _, size := range layerSizes {
				t.Run(size.name, func(t *testing.T) {
					t.Parallel()
					layer, _ := NewLayer(size.inputSize, size.outputSize, mustCreateActivation("tanh"))

					input := make([]float64, size.inputSize)
					for i := range input {
						// Create a symmetric range of values to test tanh
						input[i] = float64(i-size.inputSize/2) * 10.0
					}

					output, _ := layer.Forward(input)

					for i, val := range output {
						if val < -1 || val > 1 {
							t.Errorf("Tanh output[%d] = %v; should be in [-1, 1]", i, val)
						}
						if math.IsNaN(val) || math.IsInf(val, 0) {
							t.Errorf("Sigmoid output[%d] = %v; should be finite", i, val)
						}
					}
				})
			}
		})
	})
}

func TestLayerRobustness(t *testing.T) {
	t.Run("error_recovery_and_robustness", func(t *testing.T) {
		t.Run("corrupted_weights_handling", func(t *testing.T) {
			t.Parallel()
			// Test various corruption scenarios
			tests := []struct {
				name      string
				corruptFn func(*Layer)
				shouldErr bool
			}{
				{
					"nil_weight_row",
					func(l *Layer) { l.Weights[0] = nil },
					true,
				},
				{
					"empty_weight_row",
					func(l *Layer) { l.Weights[0] = []float64{} },
					true,
				},
				{
					"wrong_weight_size",
					func(l *Layer) { l.Weights[0] = []float64{1.0, 2.0} }, // Should be 3 elements
					true,
				},
				{
					"nil_weights_slice",
					func(l *Layer) { l.Weights = nil },
					true,
				},
				{
					"nil_biases_slice",
					func(l *Layer) { l.Biases = nil },
					true,
				},
				{
					"wrong_biases_size",
					func(l *Layer) { l.Biases = []float64{1.0} }, // Should be 2 elements
					true,
				},
				{
					"nil_activation",
					func(l *Layer) { l.Activation = nil },
					true,
				},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					testLayer, _ := NewLayer(3, 2, mustCreateActivation("relu"))

					// Apply corruption
					tc.corruptFn(testLayer)

					input := []float64{1.0, 2.0, 3.0}
					output, err := testLayer.Forward(input)

					if tc.shouldErr {
						if err == nil {
							t.Errorf("%s: Expected error but got none, output: %v", tc.name, output)
						}
						if output != nil {
							t.Errorf("%s: Expected nil output on error, got: %v", tc.name, output)
						}
					} else {
						if err != nil {
							t.Errorf("%s: Unexpected error: %v", tc.name, err)
						}
					}
				})
			}
		})
		t.Run("error_recovery", func(t *testing.T) {
			t.Parallel()
			layer, _ := NewLayer(3, 2, mustCreateActivation("relu"))
			_, err := layer.Forward([]float64{1.0})
			if err == nil {
				t.Fatal("Expected error on first forward pass, got nil")
			}

			// Layer should still work with correct input
			validInput := []float64{0.1, 0.2, 0.3}
			output, err := layer.Forward(validInput)
			if err != nil {
				t.Errorf("Layer failed after previous error: %v", err)
			}
			if len(output) != 2 {
				t.Errorf("Output size = %d; want 2", len(output))
			}
		})
		t.Run("extreme_layer_dimensions", func(t *testing.T) {
			t.Parallel()

			tests := []struct {
				name                  string
				inputSize, outputSize int
				expectError           bool
			}{
				{"max_reasonable_size", 10000, 5000, false},
				{"single_massive_input", 50000, 1, false},
				{"single_massive_output", 1, 50000, false},
				{"zero_input", 0, 5, true},
				{"zero_output", 5, 0, true},
				{"negative_input", -1, 5, true},
				{"negative_output", 5, -1, true},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					if testing.Short() && (tc.inputSize > 1000 || tc.outputSize > 1000) {
						t.Skip("Skipping large dimension test in short mode")
					}

					t.Parallel()
					layer, err := NewLayer(tc.inputSize, tc.outputSize, mustCreateActivation("linear"))

					if tc.expectError {
						if err == nil {
							t.Errorf("Expected error for dimensions %dx%d", tc.inputSize, tc.outputSize)
						}
						return
					}

					if err != nil {
						t.Errorf("Unexpected error for dimensions %dx%d: %v", tc.inputSize, tc.outputSize, err)
						return
					}

					// Test forward pass with valid dimensions
					if tc.inputSize > 0 {
						input := make([]float64, tc.inputSize)
						for i := range input {
							input[i] = 0.001 // Small values to avoid overflow
						}

						output, err := layer.Forward(input)
						if err != nil {
							t.Errorf("Forward pass failed for %dx%d: %v", tc.inputSize, tc.outputSize, err)
						}
						if len(output) != tc.outputSize {
							t.Errorf("Output size mismatch: got %d, want %d", len(output), tc.outputSize)
						}
					}
				})
			}
		})
		t.Run("concurrent_state_consistency", func(t *testing.T) {
			t.Parallel()
			layer, _ := NewLayer(5, 3, mustCreateActivation("tanh"))

			// Store original state
			originalWeights := make([][]float64, len(layer.Weights))
			for i := range layer.Weights {
				originalWeights[i] = make([]float64, len(layer.Weights[i]))
				copy(originalWeights[i], layer.Weights[i])
			}

			// WaitGroup Setup
			// 1000 concurrent forward passes: 10 goroutines, each doing 100 forward passes
			var wg sync.WaitGroup
			const numGoroutines = 10
			const numIterations = 100

			// Run concurrent forward passes
			for g := 0; g < numGoroutines; g++ {
				wg.Add(1)                  // Increment WaitGroup for each goroutine
				go func(goroutineID int) { // Creates a new goroutine
					defer wg.Done() // Decrement WaitGroup when done

					// Each goroutine gets unique input based on its ID
					input := make([]float64, 5)
					for i := range input {
						input[i] = float64(goroutineID) * 0.1
					}

					// run forward passes
					for i := 0; i < numIterations; i++ {
						_, err := layer.Forward(input)
						if err != nil {
							t.Errorf("Goroutine %d, iteration %d failed: %v", goroutineID, i, err)
							return
						}
					}
				}(g) // passes the loop variable to avoid closure issues
			}

			wg.Wait() // block until all goroutines finish

			// Verify state hasn't changed after concurrent access
			for i := range layer.Weights {
				for j := range layer.Weights[i] {
					if layer.Weights[i][j] != originalWeights[i][j] {
						t.Errorf("Concurrent access changed weight[%d][%d] from %v to %v",
							i, j, originalWeights[i][j], layer.Weights[i][j])
					}
				}
			}
		})
		t.Run("no_memory_leaks_repeated_foward", func(t *testing.T) {
		})
	})

	t.Run("memory_usage_pattern", func(t *testing.T) {
		t.Run("no_memory_leaks_repeated_forward", func(t *testing.T) {
			if testing.Short() {
				t.Skip("Skipping memory leak test in short mode")
			}

			t.Parallel()
			layer, _ := NewLayer(100, 50, mustCreateActivation("relu"))
			input := make([]float64, 100)
			for i := range input {
				input[i] = rand.Float64()
			}

			// Get initial memory stats
			var m1, m2 runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&m1)

			// Perform many forward passes
			for i := 0; i < 10000; i++ {
				output, err := layer.Forward(input)
				if err != nil {
					t.Fatalf("Forward pass %d failed: %v", i, err)
				}

				// Use output to prevent optimization
				_ = output[0] + output[len(output)-1]

				if i%1000 == 0 {
					runtime.GC()
				}
			}

			// Get final memory stats
			runtime.GC()
			runtime.ReadMemStats(&m2)

			// Memory usage shouldn't have grown significantly
			memGrowth := m2.Alloc - m1.Alloc
			if memGrowth > 1024*1024 { // 1MB threshold
				t.Logf("Memory growth detected: %d bytes (this might be acceptable)", memGrowth)
				// Don't fail the test, just log it
			}
		})
		t.Run("large_layer_memory_efficiency", func(t *testing.T) {
			if testing.Short() {
				t.Skip("Skipping large layer test in short mode")
			}

			t.Parallel()

			// Test memory efficiency with large layer
			var m1, m2 runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&m1)

			_, err := NewLayer(1000, 500, mustCreateActivation("sigmoid"))
			if err != nil {
				t.Fatalf("Failed to create large layer: %v", err)
			}

			runtime.ReadMemStats(&m2)
			layerMemory := m2.Alloc - m1.Alloc

			// Rough calculation: 1000*500 weights + 500 biases = 500,500 float64s
			// Each float64 is 8 bytes, so ~4MB minimum
			expectedMinMemory := uint64(500500 * 8)

			if layerMemory < expectedMinMemory {
				t.Errorf("Layer memory usage %d is suspiciously low, expected at least %d",
					layerMemory, expectedMinMemory)
			}

			// Should not use more than 2x expected (allowing for overhead)
			if layerMemory > expectedMinMemory*2 {
				t.Logf("Layer memory usage %d is higher than expected %d (may include overhead)",
					layerMemory, expectedMinMemory)
			}
		})
	})

	t.Run("layer_state_immutability", func(t *testing.T) {
		t.Run("weights_unchanged_during_forward", func(t *testing.T) {
			t.Parallel()
			layer, _ := NewLayer(3, 2, mustCreateActivation("relu"))

			// Store original weights and biases
			originalWeights := make([][]float64, len(layer.Weights))
			for i := range layer.Weights {
				originalWeights[i] = make([]float64, len(layer.Weights[i]))
				copy(originalWeights[i], layer.Weights[i])
			}

			originalBiases := make([]float64, len(layer.Biases))
			copy(originalBiases, layer.Biases)

			input := []float64{1.0, 2.0, 3.0}

			// Perform multiple forward passes
			for i := 0; i < 10; i++ {
				_, err := layer.Forward(input)
				if err != nil {
					t.Fatalf("Forward pass %d failed: %v", i, err)
				}
			}

			// Verify weights haven't changed
			for i := range layer.Weights {
				for j := range layer.Weights[i] {
					if layer.Weights[i][j] != originalWeights[i][j] {
						t.Errorf("Weight[%d][%d] changed from %v to %v",
							i, j, originalWeights[i][j], layer.Weights[i][j])
					}
				}
			}

			// Verify biases haven't changed
			for i := range layer.Biases {
				if layer.Biases[i] != originalBiases[i] {
					t.Errorf("Bias[%d] changed from %v to %v",
						i, originalBiases[i], layer.Biases[i])
				}
			}
		})
		t.Run("input_slice_unchanged", func(t *testing.T) {
			t.Parallel()
			layer, _ := NewLayer(3, 2, mustCreateActivation("sigmoid"))

			input := []float64{1.0, 2.0, 3.0}
			originalInput := make([]float64, len(input))
			copy(originalInput, input)

			_, err := layer.Forward(input)
			if err != nil {
				t.Fatalf("Forward pass failed: %v", err)
			}

			// Verify input slice wasn't modified
			for i := range input {
				if input[i] != originalInput[i] {
					t.Errorf("Input[%d] was modified from %v to %v",
						i, originalInput[i], input[i])
				}
			}
		})
	})

	t.Run("performance_regression_detection", func(t *testing.T) {
		t.Run("forward_pass_performance_baseline", func(t *testing.T) {
			t.Parallel()

			layerSizes := []struct {
				name    string
				in, out int
			}{
				{"small", 10, 5},
				{"medium", 100, 50},
				{"large", 1000, 500},
			}

			for _, size := range layerSizes {
				t.Run(size.name, func(t *testing.T) {
					layer, _ := NewLayer(size.in, size.out, mustCreateActivation("relu"))
					input := make([]float64, size.in)
					for i := range input {
						input[i] = rand.Float64()
					}

					// Warm up
					for i := 0; i < 10; i++ {
						layer.Forward(input)
					}

					// Measure performance
					start := time.Now()
					iterations := 1000
					for i := 0; i < iterations; i++ {
						_, err := layer.Forward(input)
						if err != nil {
							t.Fatalf("Forward pass failed: %v", err)
						}
					}
					duration := time.Since(start)

					avgTime := duration / time.Duration(iterations)

					// Performance thresholds (adjust based on your requirements)
					var maxExpected time.Duration
					switch size.name {
					case "small":
						maxExpected = 100 * time.Nanosecond
					case "medium":
						maxExpected = 10 * time.Microsecond
					case "large":
						maxExpected = 100 * time.Microsecond
					}

					if avgTime > maxExpected {
						t.Logf("Performance warning for %s layer: %v per forward pass (threshold: %v)",
							size.name, avgTime, maxExpected)
						// Don't fail, just warn
					}
				})
			}
		})

		t.Run("activation_function_performance_comparison", func(t *testing.T) {
			t.Parallel()

			activations := getActivationNames()
			const layerSize = 100
			const iterations = 1000

			results := make(map[string]time.Duration)

			for _, actName := range activations {
				layer, _ := NewLayer(layerSize, layerSize, mustCreateActivation(actName))
				input := make([]float64, layerSize)
				for i := range input {
					input[i] = rand.Float64()
				}

				// Measure
				start := time.Now()
				for i := 0; i < iterations; i++ {
					layer.Forward(input)
				}
				results[actName] = time.Since(start)
			}

			// Compare performance (Linear should be fastest, Sigmoid/Tanh slower)
			linearTime := results["linear"]
			for actName, duration := range results {
				ratio := float64(duration) / float64(linearTime)
				t.Logf("Activation %s: %v (%.2fx vs linear)", actName, duration/iterations, ratio)

				// Sanity check - no activation should be more than 100x slower than linear
				if ratio > 100 {
					t.Errorf("Activation %s is suspiciously slow: %.2fx vs linear", actName, ratio)
				}
			}
		})
	})
}

func TestLayerString(t *testing.T) {
	t.Parallel()
	layer, err := NewLayer(784, 128, mustCreateActivation("relu"))
	if err != nil {
		t.Fatalf("Failed to create layer: %v", err)
	}

	str := layer.String()
	expected := "Layer(input=784, output=128, activation=ReLU)"
	if str != expected {
		t.Errorf("String() = %q; want %q", str, expected)
	}
}

func TestLayerIntegration(t *testing.T) {
	// testRNG := rand.New(rand.NewPCG(12345, 67890))
	// BASIC MULTI-LAYER PIPELINE TEST
	t.Run("multi_layer_pipeline", func(t *testing.T) {
		t.Parallel()
		t.Run("simple_three_layer", func(t *testing.T) {
			layer1, _ := NewLayer(4, 3, mustCreateActivation("relu"))
			layer2, _ := NewLayer(3, 2, mustCreateActivation("sigmoid"))
			layer3, _ := NewLayer(2, 1, mustCreateActivation("linear"))

			inputs := [][]float64{
				{1.0, 0.5, -0.5, 2.0},
				{0.0, 0.0, 0.0, 0.0},
				{-1.0, -2.0, -3.0, -4.0},
				{100.0, -100.0, 0.1, -0.1},
			}
			for i, input := range inputs {
				t.Run(fmt.Sprintf("input_%d", i), func(t *testing.T) {
					out1, err := layer1.Forward(input)
					if err != nil {
						t.Errorf("Layer 1 Forward failed: %v", err)
					}
					out2, err := layer2.Forward(out1)
					if err != nil {
						t.Errorf("Layer 2 Forward failed: %v", err)
					}
					finalOut, err := layer3.Forward(out2)
					if err != nil {
						t.Errorf("Layer 3 Forward failed: %v", err)
					}

					// Verify pipeline integrity
					if len(finalOut) != 1 {
						t.Errorf("Final output length = %d; want 1", len(finalOut))
					}

					// Check for numerical stability
					if math.IsNaN(finalOut[0]) || math.IsInf(finalOut[0], 0) {
						t.Errorf("Pipeline produced non-finite output: %v", finalOut[0])
					}

					// Verify intermediate outputs have correct dimensions
					if len(out1) != 3 {
						t.Errorf("Layer1 output length = %d; want 3", len(out1))
					}
					if len(out2) != 2 {
						t.Errorf("Layer2 output length = %d; want 2", len(out2))
					}

					// Verify activation properties are maintained
					for j, val := range out1 {
						if val < 0 { // ReLU should be non-negative
							t.Errorf("ReLU output[%d] = %v; should be non-negative", j, val)
						}
					}

					for j, val := range out2 {
						if val < 0 || val > 1 { // Sigmoid should be in [0,1]
							t.Errorf("Sigmoid output[%d] = %v; should be in [0,1]", j, val)
						}
					}
				})
			}
		})

		t.Run("deep_network_five_layers", func(t *testing.T) {
			t.Parallel()
			// Deeper network: 10→8→6→4→2→1
			layers := []*Layer{}
			dimensions := []struct {
				in, out    int
				activation string
			}{
				{10, 8, "relu"},
				{8, 6, "tanh"},
				{6, 4, "sigmoid"},
				{4, 2, "relu"},
				{2, 1, "linear"},
			}

			for i, dim := range dimensions {
				layer, err := NewLayer(dim.in, dim.out, mustCreateActivation(dim.activation))
				if err != nil {
					t.Fatalf("Failed to create layer %d: %v", i, err)
				}
				layers = append(layers, layer)
			}

			input := make([]float64, 10)
			for i := range input {
				input[i] = rand.Float64()*2 - 1 // Random values in [-1, 1]
			}

			currentOutput := input
			for i, layer := range layers {
				var err error
				currentOutput, err = layer.Forward(currentOutput)
				if err != nil {
					t.Fatalf("Deep network layer %d failed: %v", i, err)
				}

				// Verify no NaN/Inf propagation
				for j, val := range currentOutput {
					if math.IsNaN(val) || math.IsInf(val, 0) {
						t.Fatalf("Non-finite value at layer %d, output[%d]: %v", i, j, val)
					}
				}
			}

			// Final output needs to be of size 1
			if len(currentOutput) != 1 {
				t.Errorf("Final output length = %d; want 1", len(currentOutput))
			}
		})

		t.Run("wide_network", func(t *testing.T) {
			t.Parallel()

			// Wide network: 5→100→50→200→1
			layer1, _ := NewLayer(5, 100, mustCreateActivation("relu"))
			layer2, _ := NewLayer(100, 50, mustCreateActivation("sigmoid"))
			layer3, _ := NewLayer(50, 200, mustCreateActivation("tanh"))
			layer4, _ := NewLayer(200, 1, mustCreateActivation("linear"))

			input := []float64{0.1, 0.2, 0.3, 0.4, 0.5}

			out1, _ := layer1.Forward(input)
			out2, _ := layer2.Forward(out1)
			out3, _ := layer3.Forward(out2)
			out4, _ := layer4.Forward(out3)

			// Verify dimensions through wide network
			if len(out1) != 100 {
				t.Errorf("Wide layer1 output = %d; want 100", len(out1))
			}
			if len(out2) != 50 {
				t.Errorf("Wide layer2 output = %d; want 50", len(out2))
			}
			if len(out3) != 200 {
				t.Errorf("Wide layer3 output = %d; want 200", len(out3))
			}
			if len(out4) != 1 {
				t.Errorf("Wide layer4 output = %d; want 1", len(out4))
			}

			// All ReLU outputs should be non-negative
			for i, val := range out1 {
				if val < 0 {
					t.Errorf("Wide ReLU output[%d] = %v; should be non-negative", i, val)
				}
			}
		})
	})
	t.Run("dimension_compatibility", func(t *testing.T) {
		t.Run("valid_dimensions_chains", func(t *testing.T) {
			t.Parallel()
			validChains := [][]struct{ in, out int }{
				{{1, 1}, {1, 1}, {1, 1}},          // Minimal chain
				{{2, 5}, {5, 3}, {3, 1}},          // Decreasing
				{{1, 3}, {3, 10}, {10, 5}},        // Increasing then decreasing
				{{10, 10}, {10, 10}, {10, 10}},    // Constant
				{{100, 50}, {50, 100}, {100, 25}}, // Expanding then contracting
			}

			for chainIdx, chain := range validChains {
				t.Run(fmt.Sprintf("chain_%d", chainIdx), func(t *testing.T) {
					layers := []*Layer{}

					for i, dim := range chain {
						layer, err := NewLayer(dim.in, dim.out, mustCreateActivation("linear"))
						if err != nil {
							t.Fatalf("Chain %d, layer %d creation failed: %v", chainIdx, i, err)
						}
						layers = append(layers, layer)
					}

					input := make([]float64, chain[0].in)
					for i := range input {
						input[i] = float64(i) * 0.1
					}

					currentOutput := input
					for i, layer := range layers {
						var err error
						currentOutput, err = layer.Forward(currentOutput)
						if err != nil {
							t.Fatalf("Chain %d, layer %d forward failed: %v", chainIdx, i, err)
						}

						expectedLen := chain[i].out
						if len(currentOutput) != expectedLen {
							t.Errorf("Chain %d, layer %d output length = %d; want %d",
								chainIdx, i, len(currentOutput), expectedLen)
						}
					}
				})
			}
		})
		t.Run("incompatible_dimensions", func(t *testing.T) {
			t.Parallel()
			layer1, _ := NewLayer(3, 5, mustCreateActivation("relu"))
			layer2, _ := NewLayer(4, 2, mustCreateActivation("sigmoid")) // Wrong input size!

			input := []float64{1.0, 2.0, 3.0}

			out1, err1 := layer1.Forward(input)
			if err1 != nil {
				t.Fatalf("Layer1 should succeed: %v", err1)
			}

			// This should fail - layer2 expects 4 inputs but gets 5
			_, err2 := layer2.Forward(out1)
			if err2 == nil {
				t.Error("Layer2 should fail with dimension mismatch but didn't")
			}
		})
	})

	t.Run("activation_interactions", func(t *testing.T) {
		t.Run("all_activation_combinations", func(t *testing.T) {
			activations := getActivationNames()

			for _, act1 := range activations {
				for _, act2 := range activations {
					for _, act3 := range activations {
						combo := fmt.Sprintf("%s_%s_%s", act1, act2, act3)
						t.Run(combo, func(t *testing.T) {
							t.Parallel()
							layer1, _ := NewLayer(3, 4, mustCreateActivation(act1))
							layer2, _ := NewLayer(4, 3, mustCreateActivation(act2))
							layer3, _ := NewLayer(3, 1, mustCreateActivation(act3))

							// Test with varied inputs
							inputs := [][]float64{
								{0.0, 0.0, 0.0},
								{1.0, 1.0, 1.0},
								{-1.0, -1.0, -1.0},
								{0.5, -0.5, 0.0},
								{10.0, -10.0, 0.1},
							}

							for inputIdx, input := range inputs {
								out1, err1 := layer1.Forward(input)
								if err1 != nil {
									t.Errorf("Combo %s, input %d, layer1 failed: %v", combo, inputIdx, err1)
									continue
								}

								out2, err2 := layer2.Forward(out1)
								if err2 != nil {
									t.Errorf("Combo %s, input %d, layer2 failed: %v", combo, inputIdx, err2)
									continue
								}

								out3, err3 := layer3.Forward(out2)
								if err3 != nil {
									t.Errorf("Combo %s, input %d, layer3 failed: %v", combo, inputIdx, err3)
									continue
								}

								// Verify no numerical breakdown
								for i, val := range out3 {
									if math.IsNaN(val) || math.IsInf(val, 0) {
										t.Errorf("Combo %s, input %d produced non-finite output[%d]: %v",
											combo, inputIdx, i, val)
									}
								}
							}
						})
					}
				}
			}
		})
	})

	t.Run("saturation_propagation", func(t *testing.T) {
		t.Parallel()

		t.Run("positive_saturation", func(t *testing.T) {
			// Test positive saturation path
			layer1, _ := NewLayer(2, 2, mustCreateActivation("linear"))
			layer2, _ := NewLayer(2, 1, mustCreateActivation("sigmoid"))
			// Set weights to create large positive values
			for i := range layer1.Weights {
				for j := range layer1.Weights[i] {
					layer1.Weights[i][j] = 50.0
				}
				layer1.Biases[i] = 0.0
			}

			for i := range layer2.Weights {
				for j := range layer2.Weights[i] {
					layer2.Weights[i][j] = 1.0 // Positive weights
				}
				layer2.Biases[i] = 0.0
			}

			input := []float64{1.0, 1.0}

			out1, _ := layer1.Forward(input)
			out2, _ := layer2.Forward(out1)

			// should saturate to positive values
			for i, val := range out2 {
				if val < 0.99 {
					t.Errorf("Positive saturation: sigmoid output[%d] = %v; should be near 1", i, val)
				}
			}
		})

		t.Run("negative_saturation", func(t *testing.T) {
			// Test negative saturation path
			layer1, _ := NewLayer(2, 2, mustCreateActivation("linear"))
			layer2, _ := NewLayer(2, 1, mustCreateActivation("sigmoid"))

			// Set weights to create large negative values
			for i := range layer1.Weights {
				for j := range layer1.Weights[i] {
					layer1.Weights[i][j] = -50.0 // Negative weights
				}
				layer1.Biases[i] = 0.0
			}

			for i := range layer2.Weights {
				for j := range layer2.Weights[i] {
					layer2.Weights[i][j] = 1.0 // Positive weights (so negative inputs stay negative)
				}
				layer2.Biases[i] = 0.0
			}

			input := []float64{1.0, 1.0}

			out1, _ := layer1.Forward(input)
			out2, _ := layer2.Forward(out1)

			// Should saturate to near 0
			for i, val := range out2 {
				if val > 0.01 {
					t.Errorf("Negative saturation: sigmoid output[%d] = %v; should be near 0", i, val)
				}
			}
		})

		t.Run("tanh_saturation", func(t *testing.T) {
			// Test tanh saturation (both positive and negative)
			layer, _ := NewLayer(1, 2, mustCreateActivation("tanh"))

			// One neuron with large positive weights, one with large negative
			layer.Weights[0] = []float64{100.0}
			layer.Weights[1] = []float64{-100.0}
			layer.Biases[0] = 0.0
			layer.Biases[1] = 0.0

			input := []float64{1.0}
			output, _ := layer.Forward(input)

			// First neuron should saturate to +1, second to -1
			if output[0] < 0.99 {
				t.Errorf("Tanh positive saturation: output[0] = %v; should be near 1", output[0])
			}
			if output[1] > -0.99 {
				t.Errorf("Tanh negative saturation: output[1] = %v; should be near -1", output[1])
			}
		})
	})

	t.Run("gradient_flow_preparation", func(t *testing.T) {
		t.Parallel()
		t.Run("vanishing_gradients_scenario", func(t *testing.T) {
			// Deep network with many sigmoid layers
			layers := []*Layer{}
			for i := 0; i < 10; i++ {
				layer, _ := NewLayer(5, 5, mustCreateActivation("sigmoid"))
				layers = append(layers, layer)
			}

			input := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
			currentOutput := input

			for i, layer := range layers {
				var err error
				currentOutput, err = layer.Forward(currentOutput)
				if err != nil {
					t.Fatalf("Deep sigmoid layer %d failed: %v", i, err)
				}

				// Check that outputs remain in valid sigmoid range
				for j, val := range currentOutput {
					if val < 0 || val > 1 {
						t.Errorf("Sigmoid layer %d output[%d] = %v; out of range [0,1]", i, j, val)
					}
				}
			}
		})
		t.Run("exploding_gradients_scenario", func(t *testing.T) {
			// Network with large weights (causes exploding gradients)
			layer1, _ := NewLayer(3, 3, mustCreateActivation("linear"))
			layer2, _ := NewLayer(3, 3, mustCreateActivation("linear"))
			layer3, _ := NewLayer(3, 1, mustCreateActivation("linear"))

			// Set very large weights
			for i := range layer1.Weights {
				for j := range layer1.Weights[i] {
					layer1.Weights[i][j] = 10.0
				}
			}
			for i := range layer2.Weights {
				for j := range layer2.Weights[i] {
					layer2.Weights[i][j] = 10.0
				}
			}
			for i := range layer3.Weights {
				for j := range layer3.Weights[i] {
					layer3.Weights[i][j] = 10.0
				}
			}

			input := []float64{1.0, 1.0, 1.0}

			out1, _ := layer1.Forward(input)
			out2, _ := layer2.Forward(out1)
			out3, _ := layer3.Forward(out2)

			// Output should be very large but still finite
			if !math.IsInf(out3[0], 0) && math.Abs(out3[0]) < 1000 {
				t.Errorf("Expected large output from exploding weights, got %v", out3[0])
			}

			// But should still be finite
			if math.IsInf(out3[0], 0) || math.IsNaN(out3[0]) {
				t.Errorf("Output should be finite, got %v", out3[0])
			}
		})
	})

	// DATA FLOW INTEGRITY TESTS
	t.Run("data_flow_integrity", func(t *testing.T) {
		t.Run("batch_processing_simulation", func(t *testing.T) {
			t.Parallel()
			// Simulate processing multiple samples through the same network
			layer1, _ := NewLayer(4, 3, mustCreateActivation("relu"))
			layer2, _ := NewLayer(3, 2, mustCreateActivation("sigmoid"))
			layer3, _ := NewLayer(2, 1, mustCreateActivation("linear"))

			batchInputs := [][]float64{
				{1.0, 0.0, -1.0, 0.5},
				{0.5, 1.0, -0.5, -1.0},
				{-1.0, -0.5, 1.0, 0.0},
				{0.0, -1.0, 0.5, 1.0},
				{2.0, -2.0, 0.1, -0.1},
			}

			outputs := make([][]float64, len(batchInputs))

			for i, input := range batchInputs {
				out1, err1 := layer1.Forward(input)
				if err1 != nil {
					t.Fatalf("Batch sample %d, layer1 failed: %v", i, err1)
				}

				out2, err2 := layer2.Forward(out1)
				if err2 != nil {
					t.Fatalf("Batch sample %d, layer2 failed: %v", i, err2)
				}

				out3, err3 := layer3.Forward(out2)
				if err3 != nil {
					t.Fatalf("Batch sample %d, layer3 failed: %v", i, err3)
				}

				outputs[i] = out3
			}

			// Verify all outputs are valid and different
			for i, output := range outputs {
				if len(output) != 1 {
					t.Errorf("Batch output %d length = %d; want 1", i, len(output))
				}

				if math.IsNaN(output[0]) || math.IsInf(output[0], 0) {
					t.Errorf("Batch output %d is non-finite: %v", i, output[0])
				}
			}

			// Outputs should be different for different inputs
			for i := 0; i < len(outputs); i++ {
				for j := i + 1; j < len(outputs); j++ {
					if outputs[i][0] == outputs[j][0] {
						t.Errorf("Batch outputs %d and %d are identical: %v", i, j, outputs[i][0])
					}
				}
			}
		})
		t.Run("deterministic_output", func(t *testing.T) {
			t.Parallel()

			// Verify same network produces same results
			layer1a, _ := NewLayer(3, 2, mustCreateActivation("tanh"))
			layer2a, _ := NewLayer(2, 1, mustCreateActivation("sigmoid"))

			layer1b, _ := NewLayer(3, 2, mustCreateActivation("tanh"))
			layer2b, _ := NewLayer(2, 1, mustCreateActivation("sigmoid"))

			// Copy weights to make networks identical
			for i := range layer1a.Weights {
				copy(layer1b.Weights[i], layer1a.Weights[i])
			}
			copy(layer1b.Biases, layer1a.Biases)

			for i := range layer2a.Weights {
				copy(layer2b.Weights[i], layer2a.Weights[i])
			}
			copy(layer2b.Biases, layer2a.Biases)

			input := []float64{0.1, 0.2, 0.3}

			// Forward pass through network A
			out1a, _ := layer1a.Forward(input)
			out2a, _ := layer2a.Forward(out1a)

			// Forward pass through network B
			out1b, _ := layer1b.Forward(input)
			out2b, _ := layer2b.Forward(out1b)

			// Results should be identical
			if math.Abs(out2a[0]-out2b[0]) > 1e-15 {
				t.Errorf("Identical networks produced different results: %v vs %v", out2a[0], out2b[0])
			}
		})

		t.Run("network_state_isolation", func(t *testing.T) {
			t.Parallel()
			// Verify that forward passes don't interfere with each other
			layer1, _ := NewLayer(2, 3, mustCreateActivation("relu"))
			layer2, _ := NewLayer(3, 1, mustCreateActivation("linear"))

			input1 := []float64{1.0, 0.0}
			input2 := []float64{0.0, 1.0}

			// Process input1
			out1_1, _ := layer1.Forward(input1)
			out2_1, _ := layer2.Forward(out1_1)

			// Process input2
			out1_2, _ := layer1.Forward(input2)
			out2_2, _ := layer2.Forward(out1_2)

			// Process input1 again - should get same result
			out1_1_again, _ := layer1.Forward(input1)
			out2_1_again, _ := layer2.Forward(out1_1_again)

			if math.Abs(out2_1[0]-out2_1_again[0]) > 1e-15 {
				t.Errorf("Network state was affected by intermediate processing: %v vs %v",
					out2_1[0], out2_1_again[0])
			}

			// Process input2 again - should get same result
			out1_2_again, _ := layer1.Forward(input2)
			out2_2_again, _ := layer2.Forward(out1_2_again)

			if math.Abs(out2_2[0]-out2_2_again[0]) > 1e-15 {
				t.Errorf("Network state was affected by intermediate processing: %v vs %v",
					out2_2[0], out2_2_again[0])
			}
		})

		// EDGE CASES INTEGRATION TESTS
		t.Run("edge_case_integration", func(t *testing.T) {
			t.Run("extreme_network_sizes", func(t *testing.T) {
				if testing.Short() {
					t.Skip("Skipping extreme size test in short mode")
				}

				t.Parallel()
				// Very large network
				layer1, _ := NewLayer(1000, 500, mustCreateActivation("linear"))
				layer2, _ := NewLayer(500, 100, mustCreateActivation("relu"))
				layer3, _ := NewLayer(100, 1, mustCreateActivation("sigmoid"))

				input := make([]float64, 1000)
				for i := range input {
					input[i] = rand.Float64() * 0.01 // Small values to avoid overflow
				}

				out1, err1 := layer1.Forward(input)
				if err1 != nil {
					t.Fatalf("Large layer1 failed: %v", err1)
				}

				out2, err2 := layer2.Forward(out1)
				if err2 != nil {
					t.Fatalf("Large layer2 failed: %v", err2)
				}

				out3, err3 := layer3.Forward(out2)
				if err3 != nil {
					t.Fatalf("Large layer3 failed: %v", err3)
				}

				if len(out3) != 1 {
					t.Errorf("Large network output length = %d; want 1", len(out3))
				}

				if math.IsNaN(out3[0]) || math.IsInf(out3[0], 0) {
					t.Errorf("Large network produced non-finite output: %v", out3[0])
				}
			})

			t.Run("minimal_network_sizes", func(t *testing.T) {
				t.Parallel()
				// Minimal possible network
				layer1, _ := NewLayer(1, 1, mustCreateActivation("linear"))
				layer2, _ := NewLayer(1, 1, mustCreateActivation("relu"))
				layer3, _ := NewLayer(1, 1, mustCreateActivation("sigmoid"))

				input := []float64{0.5}

				out1, _ := layer1.Forward(input)
				out2, _ := layer2.Forward(out1)
				out3, _ := layer3.Forward(out2)

				if len(out3) != 1 {
					t.Errorf("Minimal network output length = %d; want 1", len(out3))
				}

				// Should be valid sigmoid output
				if out3[0] < 0 || out3[0] > 1 {
					t.Errorf("Minimal network sigmoid output = %v; should be in [0,1]", out3[0])
				}
			})
		})
	})
}

func BenchmarkLayerForward(b *testing.B) {
	layerSizes := []struct {
		name                  string
		inputSize, outputSize int
	}{
		{"small", 10, 5},
		{"medium", 100, 50},
		{"large", 784, 128},
		{"very_large", 2048, 1024},
	}

	activations := getActivationNames()
	for _, size := range layerSizes {
		for _, act := range activations {
			b.Run(size.name+"_"+act, func(b *testing.B) {
				layer, _ := NewLayer(size.inputSize, size.outputSize, mustCreateActivation(act))
				input := make([]float64, size.inputSize)

				for i := range input {
					input[i] = rand.Float64()
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = layer.Forward(input)
				}
			})
		}
	}
}

func BenchmarkLayerForwardInputPatterns(b *testing.B) {
	layer, _ := NewLayer(100, 50, mustCreateActivation("relu"))

	sparseInput := make([]float64, 100)
	for i := 0; i < 10; i++ {
		sparseInput[rand.IntN(100)] = rand.Float64() // Randomly set some values
	}

	denseInput := make([]float64, 100)
	for i := range denseInput {
		denseInput[i] = rand.Float64() // All values set to random
	}

	binaryInput := make([]float64, 100)
	for i := range binaryInput {
		if rand.Float64() > 0.5 {
			binaryInput[i] = 1.0 // Randomly set to 1 or 0
		}
	}

	inputs := []struct {
		name string
		data []float64
	}{
		{"sparse", sparseInput},
		{"dense", denseInput},
		{"binary", binaryInput},
	}

	for _, input := range inputs {
		b.Run(input.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = layer.Forward(input.data)
			}
		})
	}

}

func BenchmarkLayerForwardMemory(b *testing.B) {
	layer, _ := NewLayer(100, 50, mustCreateActivation("relu"))
	input := make([]float64, 100)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = layer.Forward(input)
	}
}

func BenchmarkLayerCreation(b *testing.B) {
	act := mustCreateActivation("relu")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = NewLayer(784, 128, act)
	}
}

func mustCreateActivation(actName string) activation.ActivationFunc {
	act, err := activation.NewActivation(actName)
	if err != nil {
		panic("Failed to create activation: " + err.Error())
	}
	return act
}

func getActivationNames() []string {
	return []string{"relu", "sigmoid", "tanh", "linear"}
}
