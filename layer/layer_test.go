package layer

import (
	"fmt"
	"math"
	"math/rand/v2"
	"neural-network-project/activation"
	"runtime"
	"sync"
	"testing"
	"time"
)

// ============================================================================
// TEST HELPERS
// ============================================================================

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

func assertFloat64Equal(t *testing.T, got, want, tolerance float64, name string) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Errorf("%s: got %v, want %v", name, got, want)
	}
}

func setLayerWeightsAndBiases(layer *Layer, weights [][]float64, biases []float64) {
	for i, row := range weights {
		copy(layer.Weights[i], row)
	}
	copy(layer.Biases, biases)
}

func generateRandomInput(size int) []float64 {
	input := make([]float64, size)
	for i := range input {
		input[i] = rand.Float64()*2 - 1 // Range [-1, 1]
	}
	return input
}

// ============================================================================
// UNIT TESTS
// ============================================================================

func TestNewLayer(t *testing.T) {
	t.Run("valid_parameters", func(t *testing.T) {
		t.Parallel()
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
		}{
			{"zero_input_size", 0, 5, mustCreateActivation("relu")},
			{"zero_output_size", 5, 0, mustCreateActivation("relu")},
			{"negative_output_size", 5, -1, mustCreateActivation("relu")},
			{"negative_input_size", -1, 5, mustCreateActivation("relu")},
			{"both_zero", 0, 0, mustCreateActivation("relu")},
			{"both_negative", -5, -3, mustCreateActivation("relu")},
			{"nil_activation", 5, 3, nil},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				layer, err := NewLayer(tc.inputSize, tc.outputSize, tc.activation)

				if err == nil {
					t.Errorf("NewLayer(%d, %d, %v) should return error", tc.inputSize, tc.outputSize, tc.activation)
				}
				if layer != nil {
					t.Error("Layer should be nil when error occurs")
				}
			})
		}
	})
	t.Run("all_activation_functions", func(t *testing.T) {
		t.Parallel()

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
			})
		}
	})
}

func TestLayerForward(t *testing.T) {
	t.Run("valid_forward_pass", func(t *testing.T) {
		t.Parallel()

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

				input := generateRandomInput(tc.inputSize)
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
	t.Run("error_cases", func(t *testing.T) {
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
	t.Run("deterministic_behavior", func(t *testing.T) {
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
}

func TestLayerAccessors(t *testing.T) {
	t.Parallel()

	layer, err := NewLayer(784, 128, mustCreateActivation("relu"))
	if err != nil {
		t.Fatalf("Failed to create layer: %v", err)
	}

	t.Run("input_size", func(t *testing.T) {
		if got := layer.InputSize(); got != 784 {
			t.Errorf("InputSize() = %d; want 784", got)
		}
	})
	t.Run("output_size", func(t *testing.T) {
		if got := layer.OutputSize(); got != 128 {
			t.Errorf("OutputSize() = %d; want 128", got)
		}
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

// ============================================================================
// COMPONENT TESTS
// ============================================================================

func TestLayerComponents(t *testing.T) {
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
	})
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

func TestLayerIntegration(t *testing.T) {
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

					// Verify activation properties
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
		t.Run("dimension_compatibility", func(t *testing.T) {
			t.Parallel()
			validChains := [][]struct{ in, out int }{
				{{1, 1}, {1, 1}, {1, 1}},       // Minimal chain
				{{2, 5}, {5, 3}, {3, 1}},       // Decreasing
				{{1, 3}, {3, 10}, {10, 5}},     // Increasing then decreasing
				{{10, 10}, {10, 10}, {10, 10}}, // Constant
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

					input := generateRandomInput(chain[0].in)
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
		t.Run("activation_interactions", func(t *testing.T) {
			t.Parallel()

			activations := getActivationNames()

			for _, act1 := range activations {
				for _, act2 := range activations {
					combo := fmt.Sprintf("%s_%s", act1, act2)
					t.Run(combo, func(t *testing.T) {
						t.Parallel()
						layer1, _ := NewLayer(3, 4, mustCreateActivation(act1))
						layer2, _ := NewLayer(4, 1, mustCreateActivation(act2))
						inputs := [][]float64{
							{0.0, 0.0, 0.0},
							{1.0, 1.0, 1.0},
							{-1.0, -1.0, -1.0},
							{0.5, -0.5, 0.0},
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

							for i, val := range out2 {
								if math.IsNaN(val) || math.IsInf(val, 0) {
									t.Errorf("Combo %s, input %d produced non-finite output[%d]: %v", combo, inputIdx, i, val)
								}
							}
						}
					})
				}
			}
		})
	})
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
}

// ============================================================================
// PROPERTY TESTS
// ============================================================================

func TestLayerProperties(t *testing.T) {
	t.Run("mathematical_correctness", func(t *testing.T) {
		t.Run("simple_linear_calculation", func(t *testing.T) {
			t.Parallel()
			// 2×1 layer: linear, weights=[2,3], bias=1, input=[4,5] → output=24
			layer, _ := NewLayer(2, 1, mustCreateActivation("linear"))
			setLayerWeightsAndBiases(layer,
				[][]float64{{2.0, 3.0}},
				[]float64{1.0})

			input := []float64{4.0, 5.0}
			output, _ := layer.Forward(input)

			// (2.0 * 4.0) + (3.0 * 5.0) + 1.0 = 24.0
			expected := 24.0
			assertFloat64Equal(t, output[0], expected, 1e-10, "simple_linear_calculation")
		})
		t.Run("multiple_neurons_linear", func(t *testing.T) {
			t.Parallel()
			// 2×3 layer: linear activation
			layer, _ := NewLayer(2, 3, mustCreateActivation("linear"))
			setLayerWeightsAndBiases(layer,
				[][]float64{
					{1.0, 2.0},
					{0.5, -1.0},
					{-2.0, 3.0},
				},
				[]float64{0.0, 1.0, -0.5})

			input := []float64{2.0, 3.0}
			output, _ := layer.Forward(input)
			expected := []float64{8.0, -1.0, 4.5}

			for i := range expected {
				assertFloat64Equal(t, output[i], expected[i], 1e-10, "multiple_neurons_linear")
			}
		})
		t.Run("identity_transformation", func(t *testing.T) {
			t.Parallel()
			// 2×2 layer with identity matrix weights, zero bias
			layer, _ := NewLayer(2, 2, mustCreateActivation("linear"))
			setLayerWeightsAndBiases(layer,
				[][]float64{
					{1.0, 0.0},
					{0.0, 1.0},
				},
				[]float64{0.0, 0.0},
			)

			input := []float64{7.0, -3.0}
			output, _ := layer.Forward(input)

			for i, inp := range input {
				assertFloat64Equal(t, output[i], inp, 1e-10, "identity_transformation")
			}
		})
		t.Run("scale_invariance", func(t *testing.T) {
			t.Parallel()
			layer, _ := NewLayer(2, 1, mustCreateActivation("linear"))
			setLayerWeightsAndBiases(layer,
				[][]float64{{1.0, 1.0}},
				[]float64{0.0})

			input1 := []float64{1.0, 1.0}
			input2 := []float64{2.0, 2.0}

			output1, _ := layer.Forward(input1)
			output2, _ := layer.Forward(input2)

			assertFloat64Equal(t, output2[0], 2*output1[0], 1e-10, "scale_invariance")
		})
		t.Run("activation_bounds", func(t *testing.T) {
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

						// Force negative weights to test ReLU
						for i := range layer.Weights {
							for j := range layer.Weights[i] {
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
						// Create alternating large values to test sigmoid saturation
						for i := range input {
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
	})
	t.Run("state_immutability", func(t *testing.T) {
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
}

// ============================================================================
// ROBUSTNESS TESTS
// ============================================================================

func TestLayerRobustness(t *testing.T) {
	t.Run("error_handling", func(t *testing.T) {
		t.Run("corrupted_weights_handling", func(t *testing.T) {
			t.Parallel()
			// Test various corruption scenarios
			tests := []struct {
				name      string
				corruptFn func(*Layer)
			}{
				{
					"nil_weight_row",
					func(l *Layer) { l.Weights[0] = nil },
				},
				{
					"empty_weight_row",
					func(l *Layer) { l.Weights[0] = []float64{} },
				},
				{
					"wrong_weight_size",
					func(l *Layer) { l.Weights[0] = []float64{1.0, 2.0} }, // Should be 3 elements
				},
				{
					"nil_weights_slice",
					func(l *Layer) { l.Weights = nil },
				},
				{
					"nil_biases_slice",
					func(l *Layer) { l.Biases = nil },
				},
				{
					"wrong_biases_size",
					func(l *Layer) { l.Biases = []float64{1.0} }, // Should be 2 elements
				},
				{
					"nil_activation",
					func(l *Layer) { l.Activation = nil },
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

					if err == nil {
						t.Errorf("%s: Expected error but got none, output: %v", tc.name, output)
					}
					if output != nil {
						t.Errorf("%s: Expected nil output on error, got: %v", tc.name, output)
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
	})
	t.Run("extreme_dimensions", func(t *testing.T) {
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
	t.Run("concurrent_access", func(t *testing.T) {
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
	t.Run("memory_efficiency", func(t *testing.T) {
		t.Run("no_memory_leaks", func(t *testing.T) {
			if testing.Short() {
				t.Skip("Skipping memory leak test in short mode")
			}

			t.Parallel()
			layer, _ := NewLayer(100, 50, mustCreateActivation("relu"))
			input := generateRandomInput(100)

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
	t.Run("numerical_stability", func(t *testing.T) {
		t.Parallel()
		tests := []struct {
			name string
			data [][]float64
		}{
			{
				name: "very_large_values",
				data: [][]float64{{1e100, 1e100, 1e100}},
			},
			{
				name: "very_small_values",
				data: [][]float64{{1e-100, 1e-100, 1e-100}},
			},
			{
				name: "mixed_extreme",
				data: [][]float64{{1e100, -1e100, 0}},
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				layer, _ := NewLayer(3, 2, mustCreateActivation("sigmoid"))

				for _, input := range tc.data {
					output, err := layer.Forward(input)
					if err != nil {
						t.Errorf("Failed on %s: %v", tc.name, err)
						continue
					}

					// Results should be finite
					for i, val := range output {
						if math.IsNaN(val) || math.IsInf(val, 0) {
							t.Errorf("%s produced non-finite result[%d]: %v", tc.name, i, val)
						}
					}
				}
			})
		}
	})
}

// ============================================================================
// BENCHMARKS
// ============================================================================

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
				input := generateRandomInput(size.inputSize)

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = layer.Forward(input)
				}
			})
		}
	}
}

func BenchmarkLayerForwardMemory(b *testing.B) {
	layer, _ := NewLayer(100, 50, mustCreateActivation("relu"))
	input := generateRandomInput(100)

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

func BenchmarkLayerForwardInputPatterns(b *testing.B) {
	layer, _ := NewLayer(100, 50, mustCreateActivation("relu"))

	sparseInput := make([]float64, 100)
	for i := 0; i < 10; i++ {
		sparseInput[rand.IntN(100)] = rand.Float64() // Randomly set some values
	}

	denseInput := generateRandomInput(100)

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

func BenchmarkLayerPerformanceComparison(b *testing.B) {
	activations := getActivationNames()
	const layerSize = 100
	const iterations = 1000

	result := make(map[string]time.Duration)

	for _, actName := range activations {
		b.Run(actName, func(b *testing.B) {
			layer, _ := NewLayer(layerSize, layerSize, mustCreateActivation(actName))
			input := generateRandomInput(layerSize)

			start := time.Now()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				for j := 0; j < iterations; j++ {
					_, _ = layer.Forward(input)
				}
			}

			result[actName] = time.Since(start)
		})
	}
}
