package network

import (
	"fmt"
	"math"
	"math/rand/v2"
	"neural-network-project/activation"
	"neural-network-project/layer"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// ============================================================================
// TEST CONFIGURATIONS AND HELPERS
// ============================================================================

// Common test configurations used across multiple tests
var testConfigs = struct {
	tiny   NetworkConfig
	small  NetworkConfig
	medium NetworkConfig
	large  NetworkConfig
	xLarge NetworkConfig
	deep   NetworkConfig
	wide   NetworkConfig
}{
	tiny: NetworkConfig{
		LayerSizes:   []int{3, 2, 1},
		Activations:  []string{"relu", "sigmoid"},
		LearningRate: 0.01,
	},
	small: NetworkConfig{
		LayerSizes:   []int{10, 5, 1},
		Activations:  []string{"relu", "sigmoid"},
		LearningRate: 0.01,
	},
	medium: NetworkConfig{
		LayerSizes:   []int{100, 50, 25, 10},
		Activations:  []string{"relu", "relu", "sigmoid"},
		LearningRate: 0.01,
	},
	large: NetworkConfig{
		LayerSizes:   []int{784, 128, 64, 10},
		Activations:  []string{"relu", "relu", "sigmoid"},
		LearningRate: 0.001,
	},
	xLarge: NetworkConfig{
		LayerSizes:   []int{2048, 1024, 512, 256, 1},
		Activations:  []string{"relu", "relu", "relu", "sigmoid"},
		LearningRate: 0.001,
	},
	deep: NetworkConfig{
		LayerSizes:   []int{50, 48, 46, 44, 42, 40, 38, 36, 1},
		Activations:  []string{"relu", "relu", "relu", "relu", "relu", "relu", "relu", "sigmoid"},
		LearningRate: 0.001,
	},
	wide: NetworkConfig{
		LayerSizes:   []int{10, 100, 50, 5},
		Activations:  []string{"tanh", "tanh", "sigmoid"},
		LearningRate: 0.1,
	},
}

// Test helpr functions
func createAndVerifyNetwork(t *testing.T, config NetworkConfig, name string) *MLP {
	t.Helper()
	network, err := NewMLP(config)
	if err != nil {
		t.Fatalf("%s: NewMLP failed: %v", name, err)
	}

	if network == nil {
		t.Fatalf("%s: NewMLP returned nil network", name)
	}

	expectedLayerCount := len(config.LayerSizes) - 1
	if network.GetLayerCount() != expectedLayerCount {
		t.Errorf("%s: Layer count = %d; want %d", name, network.GetLayerCount(), expectedLayerCount)
	}

	if network.InputSize() != config.LayerSizes[0] {
		t.Errorf("%s: Input size = %d; want %d", name, network.InputSize(), config.LayerSizes[0])
	}

	if network.OutputSize() != config.LayerSizes[len(config.LayerSizes)-1] {
		t.Errorf("%s: Output size = %d; want %d", name, network.OutputSize(), config.LayerSizes[len(config.LayerSizes)-1])
	}

	if network.LearningRate() != config.LearningRate {
		t.Errorf("%s: Learning rate = %v; want %v", name, network.LearningRate(), config.LearningRate)
	}

	return network
}

func verifyLayerStructure(t *testing.T, network *MLP, config NetworkConfig) {
	t.Helper()

	for i := 0; i < network.GetLayerCount(); i++ {
		layer, err := network.Layer(i)
		if err != nil {
			t.Errorf("Layer(%d) returned error: %v", i, err)
			continue
		}

		expectedInput := config.LayerSizes[i]
		expectedOutput := config.LayerSizes[i+1]
		expectedActivation := config.Activations[i]

		if layer.InputSize() != expectedInput {
			t.Errorf("Layer %d: input size = %d; want %d", i, layer.InputSize(), expectedInput)
		}

		if layer.OutputSize() != expectedOutput {
			t.Errorf("Layer %d: output size = %d; want %d", i, layer.OutputSize(), expectedOutput)
		}

		actualActivation := strings.ToLower(layer.Activation.String())
		expectedActivationName := strings.ToLower(expectedActivation)
		if actualActivation != expectedActivationName {
			t.Errorf("Layer %d: activation = %s; want %s", i, actualActivation, expectedActivation)
		}
	}
}

func makeAllActivationsConfig(layerSizes []int) NetworkConfig {
	activations := getActivationNames()
	configActivations := make([]string, len(layerSizes)-1)

	// Cycle through available activations
	for i := range configActivations {
		configActivations[i] = activations[i%len(activations)]
	}

	return NetworkConfig{
		LayerSizes:   layerSizes,
		Activations:  configActivations,
		LearningRate: 0.01,
	}
}

func mustCreateNetwork(actName string) activation.ActivationFunc {
	act, err := activation.NewActivation(actName)
	if err != nil {
		panic("Failed to create activation: " + err.Error())
	}
	return act
}

func getActivationNames() []string {
	return []string{"relu", "sigmoid", "tanh", "linear"}
}

func verifyForwardOutput(t *testing.T, err error, output []float64, config NetworkConfig) {
	t.Helper()

	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if output == nil {
		t.Fatal("Output should not be nil for valid input")
	}

	// Verify output size
	expectedOutputSize := config.LayerSizes[len(config.LayerSizes)-1]
	if len(output) != expectedOutputSize {
		t.Errorf("Output size = %d; want %d", len(output), expectedOutputSize)
	}
	for i, val := range output {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			t.Errorf("Output[%d] is not finite: %v", i, val)
		}
	}

	finalActivation := config.Activations[len(config.Activations)-1]
	switch finalActivation {
	case "sigmoid":
		for i, val := range output {
			if val < 0 || val > 1 {
				t.Errorf("Sigmoid output[%d] = %v; should be in [0,1]", i, val)
			}
		}
	case "tanh":
		for i, val := range output {
			if val < -1 || val > 1 {
				t.Errorf("Tanh output[%d] = %v; should be in [-1,1]", i, val)
			}
		}
	case "relu":
		for i, val := range output {
			if val < 0 {
				t.Errorf("ReLU output[%d] = %v; should be non-negative", i, val)
			}
		}
	}
}

func generateRandomInput(size int) []float64 {
	input := make([]float64, size)
	for i := range input {
		input[i] = rand.Float64()*2 - 1 // Range [-1, 1]
	}
	return input
}

func verifyNetworkStructure(t *testing.T, network *MLP, config NetworkConfig) {
}

func verifyIntermediateResults(t *testing.T, network *MLP, input []float64, results [][]float64) {
	t.Helper()

	expectedLayerCount := network.GetLayerCount()
	if len(results) != expectedLayerCount {
		t.Errorf("Expected %d intermediate results, got %d", expectedLayerCount, len(results))
		return
	}

	// Verify dimensions
	for i, result := range results {
		layer, _ := network.Layer(i)
		expectedSize := layer.OutputSize()
		if len(result) != expectedSize {
			t.Errorf("Layer %d output size = %d, want %d", i, len(result), expectedSize)
		}

		// Check for finite values
		for j, val := range result {
			if math.IsNaN(val) || math.IsInf(val, 0) {
				t.Errorf("Layer %d output[%d] is not finite: %v", i, j, val)
			}
		}
	}

	// Verify consistency with regular Forward
	finalOutput, err := network.Forward(input)
	if err != nil {
		t.Errorf("Forward pass failed: %v", err)
		return
	}

	lastResult := results[len(results)-1]
	if len(finalOutput) != len(lastResult) {
		t.Error("Final intermediate result size doesn't match Forward output")
		return
	}

	for i := range finalOutput {
		if math.Abs(finalOutput[i]-lastResult[i]) > 1e-10 {
			t.Error("Final intermediate result doesn't match Forward output")
			break
		}
	}
}

func verifyActivationBounds(t *testing.T, layerIdx int, outputs []float64, activationType string) {
	t.Helper()

	switch activationType {
	case "relu":
		for i, val := range outputs {
			if val < 0 {
				t.Errorf("ReLU layer %d output %d is negative: %f", layerIdx, i, val)
			}
		}
	case "sigmoid":
		for i, val := range outputs {
			if val < 0 || val > 1 {
				t.Errorf("Sigmoid layer %d output %d out of bounds: %f", layerIdx, i, val)
			}
		}
	case "tanh":
		for i, val := range outputs {
			if val < -1 || val > 1 {
				t.Errorf("Tanh layer %d output %d out of bounds: %f", layerIdx, i, val)
			}
		}
	case "linear":
		// Linear has no bounds to check
	}
}

func createSimpleLinearNetwork(inputSize, outputSize int) *MLP {
	config := NetworkConfig{
		LayerSizes:   []int{inputSize, outputSize},
		Activations:  []string{"linear"},
		LearningRate: 0.01,
	}
	network, _ := NewMLP(config)
	return network
}

func setLayerWeightsAndBiases(layer *layer.Layer, weights [][]float64, biases []float64) {
	for i, weightRow := range weights {
		copy(layer.Weights[i], weightRow)
	}
	copy(layer.Biases, biases)
}

// ============================================================================
// CONSTRUCTOR AND INITIALIZATION TESTS
// ============================================================================
func TestNewNetwork(t *testing.T) {
	// ====== VALID CONFIGURATIONS ======
	t.Run("valid_configurations", func(t *testing.T) {
		t.Run("predefined_configs", func(t *testing.T) {
			testConfigs := []struct {
				name   string
				config NetworkConfig
			}{
				{"tiny", testConfigs.tiny},
				{"small", testConfigs.small},
				{"medium", testConfigs.medium},
				{"large", testConfigs.large},
				{"deep", testConfigs.deep},
				{"wide", testConfigs.wide},
			}

			for _, tc := range testConfigs {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					network := createAndVerifyNetwork(t, tc.config, tc.name)
					verifyLayerStructure(t, network, tc.config)
				})
			}
		})
		t.Run("custom_configs", func(t *testing.T) {
			t.Parallel()

			tests := []struct {
				name   string
				config NetworkConfig
			}{
				{
					name: "minimal_network",
					config: NetworkConfig{
						LayerSizes:   []int{2, 1},
						Activations:  []string{"linear"},
						LearningRate: 0.01,
					},
				},
				{
					name:   "all_activations_network",
					config: makeAllActivationsConfig([]int{8, 6, 4, 2, 1}),
				},
				{
					name: "extreme_learning_rates",
					config: NetworkConfig{
						LayerSizes:   []int{3, 2, 1},
						Activations:  []string{"relu", "sigmoid"},
						LearningRate: 1e-10, // Very small
					},
				},

				{
					name: "max_float_learning_rate",
					config: NetworkConfig{
						LayerSizes:   []int{3, 2, 1},
						Activations:  []string{"relu", "sigmoid"},
						LearningRate: math.MaxFloat64,
					},
				},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					network := createAndVerifyNetwork(t, tc.config, tc.name)
					verifyLayerStructure(t, network, tc.config)
				})
			}
		})
		t.Run("edge_cases", func(t *testing.T) {
			t.Run("very_large_network", func(t *testing.T) {
				config := NetworkConfig{
					LayerSizes:   []int{10000, 5000, 1000},
					Activations:  []string{"relu", "sigmoid"},
					LearningRate: 0.0001,
				}
				network := createAndVerifyNetwork(t, config, "very_large_network")

				if network.GetLayerCount() != 2 {
					t.Errorf("Large network layer count = %d; want 2", network.GetLayerCount())
				}
			})
			t.Run("single_neuron_layers", func(t *testing.T) {
				t.Parallel()

				config := NetworkConfig{
					LayerSizes:   []int{1, 1, 1, 1},
					Activations:  []string{"relu", "sigmoid", "tanh"},
					LearningRate: 0.01,
				}

				network := createAndVerifyNetwork(t, config, "single_neuron_layers")
				for i := 0; i < network.GetLayerCount(); i++ {
					layer, _ := network.Layer(i)
					if layer.OutputSize() != 1 {
						t.Errorf("Layer %d should have 1 neuron, got %d", i, layer.OutputSize())
					}
				}
			})
		})
		t.Run("all_activation_functions", func(t *testing.T) {
			t.Parallel()

			activationNames := getActivationNames()
			for _, actName := range activationNames {
				t.Run(actName, func(t *testing.T) {
					t.Parallel()

					config := NetworkConfig{
						LayerSizes:   []int{5, 3, 1},
						Activations:  []string{actName, actName},
						LearningRate: 0.01,
					}
					network := createAndVerifyNetwork(t, config, "all_"+actName)

					for i := 0; i < network.GetLayerCount(); i++ {
						layer, _ := network.Layer(i)
						if strings.ToLower(layer.Activation.String()) != actName {
							t.Errorf("Layer %d activation = %s; want %s",
								i, layer.Activation.String(), actName)
						}
					}
				})
			}
		})
	})
	// ===== INVALID CONFIGURATIONS =====
	t.Run("invalid_configurations", func(t *testing.T) {
		tests := []struct {
			name    string
			config  NetworkConfig
			wantErr string
		}{
			{
				"empty_layer_sizes",
				NetworkConfig{
					LayerSizes:   []int{},
					Activations:  []string{"relu"},
					LearningRate: 0.01,
				},
				"network must have at least 2 layers",
			},
			{
				"single_layer_size",
				NetworkConfig{
					LayerSizes:   []int{3},
					Activations:  []string{"relu"},
					LearningRate: 0.01,
				},
				"network must have at least 1 trainable layer",
			},
			{
				"zero_middle_layer",
				NetworkConfig{
					LayerSizes:   []int{3, 0, 1}, // Zero in middle
					Activations:  []string{"relu", "sigmoid"},
					LearningRate: 0.01,
				},
				"layer size must be positive",
			},
			{
				"empty_activations",
				NetworkConfig{
					LayerSizes:   []int{3, 2, 1},
					Activations:  []string{}, // Empty
					LearningRate: 0.01,
				},
				"number of activations must be one less than layer sizes",
			},
			{
				"activation_count_mismatch_too_few",
				NetworkConfig{
					LayerSizes:   []int{3, 2, 1},
					Activations:  []string{"relu"},
					LearningRate: 0.01,
				},
				"number of activations must be one less than layer sizes",
			},
			{
				"activation_count_mismatch_too_many",
				NetworkConfig{
					LayerSizes:   []int{3, 2},
					Activations:  []string{"relu", "sigmoid", "tanh"},
					LearningRate: 0.01,
				},
				"number of activations must be one less than layer sizes",
			},
			{
				"zero_learning_rate",
				NetworkConfig{
					LayerSizes:   []int{3, 2, 1},
					Activations:  []string{"relu", "sigmoid"},
					LearningRate: 0.0,
				},
				"learning rate must be positive",
			},
			{
				"negative_learning_rate",
				NetworkConfig{
					LayerSizes:   []int{3, 2, 1},
					Activations:  []string{"relu", "sigmoid"},
					LearningRate: -0.01,
				},
				"learning rate must be positive",
			},
			{
				"nan_learning_rate",
				NetworkConfig{
					LayerSizes:   []int{3, 2, 1},
					Activations:  []string{"relu", "sigmoid"},
					LearningRate: math.NaN(),
				},
				"learning rate must be positive",
			},
			{
				"infinite_learning_rate",
				NetworkConfig{
					LayerSizes:   []int{3, 2, 1},
					Activations:  []string{"relu", "sigmoid"},
					LearningRate: math.Inf(1),
				},
				"learning rate must be positive",
			},
			{
				"negative_layer_size",
				NetworkConfig{
					LayerSizes:   []int{-3, 1},
					Activations:  []string{"relu"},
					LearningRate: 0.01,
				},
				"layer size must be positive",
			},
			{
				"invalid_activation",
				NetworkConfig{
					LayerSizes:   []int{3, 2},
					Activations:  []string{"invalid_activation"},
					LearningRate: 0.01,
				},
				"invalid activation function",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				network, err := NewMLP(tc.config)

				if err == nil {
					t.Errorf("NewNetwork(%+v) should return error containing %q", tc.config, tc.wantErr)
				}

				if network != nil {
					t.Errorf("Network should be nil for invalid configuration: %v", tc.config)
				}

				if err != nil && tc.wantErr != "" {
					if !strings.Contains(err.Error(), tc.wantErr) {
						t.Errorf("Error message %q should contain %q", err.Error(), tc.wantErr)
					}
				}
			})
		}
	})
	// ===== BOUNDARY CASES =====
	t.Run("boundary_cases", func(t *testing.T) {
		t.Run("extreme_but_valid_networks", func(t *testing.T) {
			t.Parallel()

			extremeConfigs := []struct {
				name   string
				config NetworkConfig
			}{
				{
					name: "very_deep_narrow",
					config: NetworkConfig{
						LayerSizes:   []int{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1}, // 10 layers
						Activations:  []string{"relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "sigmoid"},
						LearningRate: 0.001,
					},
				},
				{
					name: "funnel_architecture",
					config: NetworkConfig{
						LayerSizes:   []int{1000, 100, 10, 1},
						Activations:  []string{"relu", "relu", "sigmoid"},
						LearningRate: 0.01,
					},
				},
				{
					name: "expanding_architecture",
					config: NetworkConfig{
						LayerSizes:   []int{1, 10, 100, 1000},
						Activations:  []string{"relu", "relu", "linear"},
						LearningRate: 0.001,
					},
				},
				{
					name: "hourglass_architecture",
					config: NetworkConfig{
						LayerSizes:   []int{100, 50, 10, 50, 100},
						Activations:  []string{"relu", "relu", "relu", "sigmoid"},
						LearningRate: 0.01,
					},
				},
			}

			for _, tc := range extremeConfigs {
				t.Run(tc.name, func(t *testing.T) {
					network := createAndVerifyNetwork(t, tc.config, tc.name)
					verifyLayerStructure(t, network, tc.config)
					t.Logf("%s created successfully with %d layers", tc.name, network.GetLayerCount())
				})
			}
		})
		t.Run("mixed_activation_patterns", func(t *testing.T) {
			t.Parallel()

			mixedConfigs := []struct {
				name   string
				config NetworkConfig
			}{
				{
					name: "alternating_relu_sigmoid",
					config: NetworkConfig{
						LayerSizes:   []int{10, 8, 6, 4, 2},
						Activations:  []string{"relu", "sigmoid", "relu", "sigmoid"},
						LearningRate: 0.01,
					},
				},
				{
					name: "all_different",
					config: NetworkConfig{
						LayerSizes:   []int{5, 4, 3, 2, 1},
						Activations:  []string{"relu", "sigmoid", "tanh", "linear"},
						LearningRate: 0.01,
					},
				},
				{
					name: "linear_sandwich",
					config: NetworkConfig{
						LayerSizes:   []int{10, 20, 30, 20, 10},
						Activations:  []string{"linear", "relu", "relu", "linear"},
						LearningRate: 0.01,
					},
				},
			}

			for _, tc := range mixedConfigs {
				t.Run(tc.name, func(t *testing.T) {
					network := createAndVerifyNetwork(t, tc.config, tc.name)
					verifyLayerStructure(t, network, tc.config)
				})
			}
		})
	})
}

// ============================================================================
// FORWARD PROPAGATION TESTS
// ============================================================================
func TestNetworkForward(t *testing.T) {
	t.Run("valid_forward_pass", func(t *testing.T) {
		t.Run("standard_inputs", func(t *testing.T) {
			t.Parallel()

			tests := []struct {
				name        string
				config      NetworkConfig
				input       []float64
				description string
			}{
				{
					name:        "tiny_network_mixed_input",
					config:      testConfigs.tiny,
					input:       []float64{1.0, 0.5, -0.5},
					description: "mixed positive/negative values",
				},
				{
					name: "minimal_network",
					config: NetworkConfig{
						LayerSizes:   []int{2, 1},
						Activations:  []string{"linear"},
						LearningRate: 0.01,
					},
					input:       []float64{2.0, 3.0},
					description: "simple linear network",
				},
				{
					name:        "small_network_zero_input",
					config:      testConfigs.small,
					input:       make([]float64, 10), // zeros
					description: "all zeros input",
				},
				{
					name: "sigmoid_saturation_test",
					config: NetworkConfig{
						LayerSizes:   []int{2, 1},
						Activations:  []string{"sigmoid"},
						LearningRate: 0.01,
					},
					input:       []float64{1000.0, -1000.0},
					description: "extreme values for sigmoid",
				},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()

					network := createAndVerifyNetwork(t, tc.config, tc.name)
					output, err := network.Forward(tc.input)

					verifyForwardOutput(t, err, output, tc.config)
					t.Logf("%s: %s -> output: %v", tc.name, tc.description, output)
				})
			}
		})
		t.Run("using_predefined_networks", func(t *testing.T) {
			t.Parallel()

			tests := []struct {
				name   string
				config NetworkConfig
			}{
				{"tiny", testConfigs.tiny},
				{"small", testConfigs.small},
				{"medium", testConfigs.medium},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()

					network := createAndVerifyNetwork(t, tc.config, tc.name)
					input := generateRandomInput(tc.config.LayerSizes[0])
					output, err := network.Forward(input)
					verifyForwardOutput(t, err, output, tc.config)
				})
			}
		})
	})
	t.Run("forward_error_cases", func(t *testing.T) {
		network := createAndVerifyNetwork(t, testConfigs.tiny, "error_test_network")

		tests := []struct {
			name    string
			input   []float64
			wantErr string
		}{
			{
				"nil_input",
				nil,
				"input cannot be nil",
			},
			{
				"empty_input",
				[]float64{},
				"input size mismatch: network input size 3, input vector size 0",
			},
			{
				"wrong_input_size",
				[]float64{1.0, 2.0},
				"input size mismatch: network input size 3, input vector size 2",
			},
			{
				"wrong_input_size_large",
				[]float64{1.0, 2.0, 3.0, 4.0},
				"input size mismatch: network input size 3, input vector size 4",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				output, err := network.Forward(tc.input)

				if err == nil {
					t.Errorf("Forward(%v) should return error", tc.input)
				}

				if output != nil {
					t.Error("Output should be nil on error")
				}
				if err != nil && !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("Error message %q should contain %q", err.Error(), tc.wantErr)
				}
			})
		}
	})
	t.Run("special_float_values", func(t *testing.T) {
		network := createAndVerifyNetwork(t, testConfigs.tiny, "special_float_test")

		tests := []struct {
			name  string
			input []float64
		}{
			{"with_nan", []float64{1.0, math.NaN(), 3.0}},
			{"with_inf", []float64{1.0, math.Inf(1), 3.0}},
			{"with_neg_inf", []float64{1.0, math.Inf(-1), 3.0}},
			{"all_special", []float64{math.NaN(), math.Inf(1), math.Inf(-1)}},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				output, err := network.Forward(tc.input)

				if err != nil {
					t.Logf("Network returned error for %s: %v", tc.name, err)
				} else if output != nil {
					t.Logf("Network handled %s, output: %v", tc.name, output)
				}
			})
		}
	})
	t.Run("mathematical_correctness", func(t *testing.T) {
		t.Run("zero_input_bias_only", func(t *testing.T) {
			// Test that zero inputs produce predictable outputs based on biases
			network := createSimpleLinearNetwork(2, 1)

			layer, _ := network.Layer(0)
			setLayerWeightsAndBiases(layer,
				[][]float64{{5.0, 10.0}}, // Weights don't matter with zero input
				[]float64{3.0})           // Only bias should affect output

			input := []float64{0.0, 0.0}
			output, _ := network.Forward(input)

			// With zero input, output should equal bias
			if math.Abs(output[0]-3.0) > 1e-10 {
				t.Errorf("Zero input test: got %v, want 3.0", output[0])
			}
		})
		t.Run("known_linear_calculation", func(t *testing.T) {
			t.Parallel()

			network := createSimpleLinearNetwork(2, 1)

			layer, _ := network.Layer(0)
			setLayerWeightsAndBiases(layer,
				[][]float64{{2.0, 3.0}}, // Weights for
				[]float64{1.0},          // Bias
			)

			testCases := []struct {
				input    []float64
				expected float64
			}{
				{[]float64{4.0, 5.0}, 24.0}, // 2*4 + 3*5 + 1 = 24
				{[]float64{1.0, 0.0}, 3.0},  // 2*1 + 3*0 + 1 = 3
				{[]float64{0.0, 1.0}, 4.0},  // 2*0 + 3*1 + 1 = 4
				{[]float64{-1.0, 2.0}, 5.0}, // 2*(-1) + 3*2 + 1 = 5
			}

			for _, tc := range testCases {
				output, err := network.Forward(tc.input)
				if err != nil {
					t.Fatalf("Forward failed: %v", err)
				}

				if math.Abs(output[0]-tc.expected) > 1e-10 {
					t.Errorf("Input %v: output = %v; want %v",
						tc.input, output[0], tc.expected)
				}
			}
		})
		t.Run("multi_layer_calculation", func(t *testing.T) {
			t.Parallel()

			config := NetworkConfig{
				LayerSizes:   []int{2, 2, 1},
				Activations:  []string{"linear", "linear"},
				LearningRate: 0.01,
			}
			network := createAndVerifyNetwork(t, config, "multi_layer_test")

			// Layer 1: 2→2
			layer1, _ := network.Layer(0)
			setLayerWeightsAndBiases(layer1,
				[][]float64{{1.0, 2.0}, {3.0, 4.0}}, // Weights for 2 neurons
				[]float64{0.0, 0.0},                 // Biases for 2 neurons
			)

			// Layer 2: 2→1
			layer2, _ := network.Layer(1)
			setLayerWeightsAndBiases(layer2,
				[][]float64{{0.5, 1.5}}, // Weights for 1 neuron
				[]float64{0.0},          // Bias for 1 neuron
			)

			// Test with input [1.0, 2.0]
			input := []float64{1.0, 2.0}
			output, _ := network.Forward(input)

			// Manual calculation:
			// Layer 1 output: [1*1+2*2, 3*1+4*2] = [5, 11]
			// Layer 2 output: 0.5*5 + 1.5*11 = 2.5 + 16.5 = 19
			expected := 19.0

			if math.Abs(output[0]-expected) > 1e-10 {
				t.Errorf("Multi-layer calculation: got %v, want %v", output[0], expected)
			}
		})
		t.Run("activation_specific_calculations", func(t *testing.T) {
			t.Parallel()
			activationTests := []struct {
				name         string
				activation   string
				weights      float64
				bias         float64
				input        float64
				expectedFunc func(float64) float64
			}{
				{
					"relu_negative",
					"relu",
					1.0, -5.0, 2.0, // 1*2 + (-5) = -3, ReLU(-3) = 0
					func(x float64) float64 { return 0.0 },
				},
				{
					"relu_positive",
					"relu",
					1.0, 5.0, 2.0, // 1*2 + 5 = 7, ReLU(7) = 7
					func(x float64) float64 { return 7.0 },
				},
				{
					"sigmoid_zero",
					"sigmoid",
					1.0, 0.0, 0.0, // sigmoid(0) = 0.5
					func(x float64) float64 { return 0.5 },
				},
				{
					"tanh_zero",
					"tanh",
					1.0, 0.0, 0.0, // tanh(0) = 0
					func(x float64) float64 { return 0.0 },
				},
			}

			for _, tc := range activationTests {
				t.Run(tc.name, func(t *testing.T) {
					config := NetworkConfig{
						LayerSizes:   []int{1, 1},
						Activations:  []string{tc.activation},
						LearningRate: 0.01,
					}

					network := createAndVerifyNetwork(t, config, tc.name)
					layer, _ := network.Layer(0)
					setLayerWeightsAndBiases(layer,
						[][]float64{{tc.weights}}, // Single weight
						[]float64{tc.bias},        // Single bias
					)

					output, _ := network.Forward([]float64{tc.input})
					expected := tc.expectedFunc(tc.input)

					tolerance := 1e-10
					if tc.activation == "sigmoid" {
						tolerance = 1e-6 // Sigmoid needs slightly more tolerance
					}
					if math.Abs(output[0]-expected) > tolerance {
						t.Errorf("%s calculation: got %v, want %v", tc.name, output[0], expected)
					}
				})
			}
		})
		t.Run("advanced_mathematical_properties", func(t *testing.T) {
			t.Parallel()

			t.Run("sigmoid_mathematical_correctness", func(t *testing.T) {
				network := createSimpleLinearNetwork(1, 1)
				// Change to sigmoid
				config := NetworkConfig{
					LayerSizes:   []int{1, 1},
					Activations:  []string{"sigmoid"},
					LearningRate: 0.01,
				}
				network, _ = NewMLP(config)

				layer, _ := network.Layer(0)
				setLayerWeightsAndBiases(layer,
					[][]float64{{1.0}},
					[]float64{0.0})

				tests := []struct {
					input    float64
					expected float64
					desc     string
				}{
					{0.0, 0.5, "sigmoid(0) = 0.5"},
					{math.Log(3), 0.75, "sigmoid(ln(3)) = 0.75"},
					{-math.Log(3), 0.25, "sigmoid(-ln(3)) = 0.25"},
					{10.0, 0.9999, "sigmoid(large positive) ≈ 1"},
					{-10.0, 0.0001, "sigmoid(large negative) ≈ 0"},
				}

				for _, tc := range tests {
					output, _ := network.Forward([]float64{tc.input})
					if math.Abs(output[0]-tc.expected) > 0.001 {
						t.Errorf("%s: got %v, want %v", tc.desc, output[0], tc.expected)
					}
				}
			})
			t.Run("tanh_mathematical_correctness", func(t *testing.T) {
				config := NetworkConfig{
					LayerSizes:   []int{1, 1},
					Activations:  []string{"tanh"},
					LearningRate: 0.01,
				}
				network := createAndVerifyNetwork(t, config, "tanh_calc_correctness_test")

				layer, _ := network.Layer(0)
				setLayerWeightsAndBiases(layer,
					[][]float64{{1.0}},
					[]float64{0.0})

				tests := []struct {
					input    float64
					expected float64
					desc     string
				}{
					{0.0, 0.0, "tanh(0) = 0"},
					{math.Log(3) / 2, 0.5, "tanh(ln(3)/2) = 0.5"},
					{-math.Log(3) / 2, -0.5, "tanh(-ln(3)/2) = -0.5"},
					{5.0, 0.9999, "tanh(large positive) ≈ 1"},
					{-5.0, -0.9999, "tanh(large negative) ≈ -1"},
					{1.0, 0.7616, "tanh(1) ≈ 0.7616"},
				}

				for _, tc := range tests {
					output, _ := network.Forward([]float64{tc.input})
					if math.Abs(output[0]-tc.expected) > 0.001 {
						t.Errorf("%s: got %v, want %v", tc.desc, output[0], tc.expected)
					}
				}
			})

			t.Run("activation_composition", func(t *testing.T) {
				config := NetworkConfig{
					LayerSizes:   []int{1, 2, 1},
					Activations:  []string{"relu", "sigmoid"},
					LearningRate: 0.01,
				}
				network := createAndVerifyNetwork(t, config, "activation_composition_test")

				// Set up first layer (ReLU)
				layer1, _ := network.Layer(0)
				setLayerWeightsAndBiases(layer1,
					[][]float64{{1.0}, {-1.0}}, // First neuron: x, Second neuron: -x
					[]float64{0.0, 0.0})

				// Set up second layer (Sigmoid)
				layer2, _ := network.Layer(1)
				setLayerWeightsAndBiases(layer2,
					[][]float64{{1.0, 1.0}}, // Sum both inputs
					[]float64{0.0})

				// Test cases
				tests := []struct {
					input    float64
					expected float64
					desc     string
				}{
					{2.0, 0.88, "positive input: ReLU(2) + ReLU(-2) = 2 + 0 = 2, sigmoid(2) ≈ 0.88"},
					{-2.0, 0.88, "negative input: ReLU(-2) + ReLU(2) = 0 + 2 = 2, sigmoid(2) ≈ 0.88"},
					{0.0, 0.5, "zero input: ReLU(0) + ReLU(0) = 0 + 0 = 0, sigmoid(0) = 0.5"},
				}

				for _, tc := range tests {
					output, _ := network.Forward([]float64{tc.input})
					if math.Abs(output[0]-tc.expected) > 0.02 { // Slightly more tolerance for composition
						t.Errorf("%s: got %v, want %v", tc.desc, output[0], tc.expected)
					}
				}
			})
		})

		t.Run("numerical_stability_edge_cases", func(t *testing.T) {
			t.Parallel()

			// Test numerical stability with extreme values
			config := NetworkConfig{
				LayerSizes:   []int{2, 2, 1},
				Activations:  []string{"sigmoid", "tanh"},
				LearningRate: 0.01,
			}
			network := createAndVerifyNetwork(t, config, "numerical_stability_test")

			// Set weights to create potential overflow scenarios
			layer1, _ := network.Layer(0)
			setLayerWeightsAndBiases(layer1,
				[][]float64{{100.0, 100.0}, {-100.0, -100.0}},
				[]float64{0.0, 0.0})

			layer2, _ := network.Layer(1)
			setLayerWeightsAndBiases(layer2,
				[][]float64{{1.0, 1.0}},
				[]float64{0.0})

			// These inputs could cause overflow in naive implementations
			inputs := [][]float64{
				{10.0, 10.0},
				{-10.0, -10.0},
				{100.0, -100.0},
				{1e-100, 1e100}, // Mix of very small and very large
			}

			for _, input := range inputs {
				output, err := network.Forward(input)
				if err != nil {
					t.Errorf("Forward failed for input %v: %v", input, err)
					continue
				}

				// Output should be finite and within bounds
				if math.IsNaN(output[0]) || math.IsInf(output[0], 0) {
					t.Errorf("Numerical instability for input %v: got %v", input, output[0])
				}

				// tanh output should be in [-1, 1]
				if output[0] < -1 || output[0] > 1 {
					t.Errorf("tanh output out of bounds for input %v: got %v", input, output[0])
				}
			}
		})
	})
}

func TestNetworkIntermediateResults(t *testing.T) {
	t.Run("basic_functionality", func(t *testing.T) {
		config := NetworkConfig{
			LayerSizes:   []int{3, 4, 2, 1},
			Activations:  []string{"relu", "sigmoid", "tanh"},
			LearningRate: 0.01,
		}
		network := createAndVerifyNetwork(t, config, "intermediate_results_test")

		input := []float64{1.0, -0.5, 0.5}
		results, err := network.ForwardWithIntermediateResults(input)

		verifyIntermediateResults(t, network, input, results)
		if err != nil {
			t.Fatalf("ForwardWithIntermediateResults failed: %v", err)
		}
	})
	t.Run("activation_bounds_verification", func(t *testing.T) {
		activationConfigs := []struct {
			name       string
			config     NetworkConfig
			activation string
		}{
			{
				"relu_outputs",
				NetworkConfig{
					LayerSizes:   []int{2, 3, 1},
					Activations:  []string{"relu", "relu"},
					LearningRate: 0.01,
				},
				"relu",
			},
			{
				"sigmoid_outputs",
				NetworkConfig{
					LayerSizes:   []int{2, 3, 1},
					Activations:  []string{"sigmoid", "sigmoid"},
					LearningRate: 0.01,
				},
				"sigmoid",
			},
			{
				"tanh_outputs",
				NetworkConfig{
					LayerSizes:   []int{2, 3, 1},
					Activations:  []string{"tanh", "tanh"},
					LearningRate: 0.01,
				},
				"tanh",
			},
		}

		for _, tc := range activationConfigs {
			t.Run(tc.name, func(t *testing.T) {
				network := createAndVerifyNetwork(t, tc.config, tc.name)

				// Test with various inputs
				inputs := [][]float64{
					{1.0, -1.0},
					{100.0, -100.0},
					{0.0, 0.0},
				}

				for _, input := range inputs {
					results, _ := network.ForwardWithIntermediateResults(input)

					for i, result := range results {
						verifyActivationBounds(t, i, result, tc.activation)
					}
				}
			})
		}
	})
	t.Run("error_handling", func(t *testing.T) {
		network := createAndVerifyNetwork(t, testConfigs.tiny, "error_handling_network")

		tests := []struct {
			name    string
			input   []float64
			wantErr string
		}{
			{
				"nil_input",
				nil,
				"input cannot be nil",
			},
			{
				"wrong_input_size",
				[]float64{1.0, 2.0},
				"input size mismatch",
			},
			{
				"empty_input",
				[]float64{},
				"input size mismatch",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				results, err := network.ForwardWithIntermediateResults(tc.input)

				if err == nil {
					t.Errorf("Expected error for %s", tc.name)
				}

				if results != nil {
					t.Errorf("Results should be nil for invalid input")
				}

				if err != nil && !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("Error %q should contain %q", err.Error(), tc.wantErr)
				}
			})
		}
	})
	t.Run("data_flow_tracking", func(t *testing.T) {
		config := NetworkConfig{
			LayerSizes:   []int{1, 2, 1},
			Activations:  []string{"linear", "linear"},
			LearningRate: 0.01,
		}
		network := createAndVerifyNetwork(t, config, "data_flow_test")

		layer1, _ := network.Layer(0)
		setLayerWeightsAndBiases(layer1,
			[][]float64{{2.0}, {3.0}},
			[]float64{0.0, 0.0})

		layer2, _ := network.Layer(1)
		setLayerWeightsAndBiases(layer2,
			[][]float64{{1.0, 1.0}},
			[]float64{0.0})

		input := []float64{1.0}
		results, _ := network.ForwardWithIntermediateResults(input)

		if len(results[0]) != 2 || results[0][0] != 2.0 || results[0][1] != 3.0 {
			t.Errorf("Layer 1 output incorrect: got %v, want [2, 3]", results[0])
		}

		if len(results[1]) != 1 || results[1][0] != 5.0 {
			t.Errorf("Layer 2 output incorrect: got %v, want [5]", results[1])
		}
	})
	t.Run("mixed_activation_flow", func(t *testing.T) {
		config := NetworkConfig{
			LayerSizes:   []int{1, 2, 2, 1},
			Activations:  []string{"relu", "sigmoid", "tanh"},
			LearningRate: 0.01,
		}
		network := createAndVerifyNetwork(t, config, "mixed_activation_flow_test")

		// Set up predictable weights
		layer1, _ := network.Layer(0)
		setLayerWeightsAndBiases(layer1,
			[][]float64{{1.0}, {-1.0}},
			[]float64{0.0, 0.0})

		inputs := []float64{-2.0, 0.0, 2.0}

		for _, input := range inputs {
			results, _ := network.ForwardWithIntermediateResults([]float64{input})

			relu1 := results[0][0]
			relu2 := results[0][1]

			// Verify ReLU behavior
			if input > 0 {
				if relu1 != input || relu2 != 0 {
					t.Errorf("ReLU layer failed for input %v: got [%v, %v], want [%v, 0]",
						input, relu1, relu2, input)
				}
			} else if input < 0 {
				if relu1 != 0 || relu2 != -input {
					t.Errorf("ReLU layer failed for input %v: got [%v, %v], want [0, %v]",
						input, relu1, relu2, -input)
				}
			} else {
				if relu1 != 0 || relu2 != 0 {
					t.Errorf("ReLU layer failed for input %v: got [%v, %v], want [0, 0]",
						input, relu1, relu2)
				}
			}

			// Verify activation bounds
			verifyActivationBounds(t, 1, results[1], "sigmoid")
			verifyActivationBounds(t, 2, results[2], "tanh")

		}
	})
	t.Run("consistency_with_forward", func(t *testing.T) {
		configs := []NetworkConfig{
			{LayerSizes: []int{5, 3, 1}, Activations: []string{"relu", "sigmoid"}, LearningRate: 0.01},
			{LayerSizes: []int{2, 4, 4, 2}, Activations: []string{"tanh", "relu", "sigmoid"}, LearningRate: 0.01},
			{LayerSizes: []int{10, 5}, Activations: []string{"linear"}, LearningRate: 0.01},
		}

		for i, config := range configs {
			configName := fmt.Sprintf("config_%d", i)
			t.Run(configName, func(t *testing.T) {
				network := createAndVerifyNetwork(t, config, configName)
				input := generateRandomInput(config.LayerSizes[0])

				forwardOutput, _ := network.Forward(input)
				intermediateResults, _ := network.ForwardWithIntermediateResults(input)

				lastIntermediate := intermediateResults[len(intermediateResults)-1]

				if len(forwardOutput) != len(lastIntermediate) {
					t.Errorf("Output size mismatch for config %v", config)
				}

				for j := range forwardOutput {
					if math.Abs(forwardOutput[j]-lastIntermediate[j]) > 1e-10 {
						t.Errorf("Outputs don't match for config %v: Forward=%v, Intermediate=%v",
							config, forwardOutput[j], lastIntermediate[j])
					}
				}
			})
		}
	})
}

// ============================================================================
// ACCESSOR AND UTILITY TESTS
// ============================================================================
func TestNetworkAccessors(t *testing.T) {
	network := createAndVerifyNetwork(t, testConfigs.tiny, "accessor_test")
	t.Run("Layer", func(t *testing.T) {
		for i := 0; i < network.GetLayerCount(); i++ {
			layer, err := network.Layer(i)
			if err != nil {
				t.Errorf("Layer(%d) returned error: %v", i, err)
			}

			if layer == nil {
				t.Errorf("Layer(%d) returned nil layer", i)
			}
		}

		// Test invalid indices
		for _, idx := range []int{-1, network.GetLayerCount(), 100} {
			layer, err := network.Layer(idx)
			if err == nil {
				t.Errorf("Layer(%d) should return error", idx)
			}
			if layer != nil {
				t.Errorf("Layer(%d) should return nil layer", idx)
			}
		}
	})
	t.Run("Layers", func(t *testing.T) {
		layers := network.Layers()
		if layers == nil {
			t.Error("Layers() returned nil")
		}
		if len(layers) != network.GetLayerCount() {
			t.Errorf("Layers() returned %d layers, want %d", len(layers), network.GetLayerCount())
		}

		// Verify the layers match what Layer(i) returns
		for i, layer := range layers {
			expectedLayer, _ := network.Layer(i)
			if layer != expectedLayer {
				t.Errorf("Layers()[%d] doesn't match Layer(%d)", i, i)
			}
		}
	})
	t.Run("Properties", func(t *testing.T) {
		tests := []struct {
			name string
			got  interface{}
			want interface{}
		}{
			{"GetLayerCount", network.GetLayerCount(), 2},
			{"InputSize", network.InputSize(), 3},
			{"OutputSize", network.OutputSize(), 1},
			{"LearningRate", network.LearningRate(), 0.01},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				if tc.got != tc.want {
					t.Errorf("%s() = %v, want %v", tc.name, tc.got, tc.want)
				}
			})
		}
	})
}

func TestNetworkString(t *testing.T) {
	tests := []struct {
		name     string
		config   NetworkConfig
		contains []string
	}{
		{
			"simple_network",
			testConfigs.tiny,
			[]string{"MLP", "3", "2", "1", "ReLU", "Sigmoid"},
		},
		{
			"deep_network",
			NetworkConfig{
				LayerSizes:   []int{4, 3, 2, 1},
				Activations:  []string{"relu", "relu", "sigmoid"},
				LearningRate: 0.001,
			},
			[]string{"MLP", "4", "3", "2", "1", "ReLU", "Sigmoid"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			network := createAndVerifyNetwork(t, tc.config, tc.name)
			str := network.String()

			if str == "" {
				t.Error("String() should not be empty")
			}

			for _, substr := range tc.contains {
				if !strings.Contains(str, substr) {
					t.Errorf("String() should contain %q, got: %s", substr, str)
				}
			}
		})
	}
}

// ============================================================================
// COMPONENT AND INTEGRATION TESTS
// ============================================================================

func TestNetworkComponent(t *testing.T) {
	t.Run("deterministic_behavior", func(t *testing.T) {
		t.Parallel()
		config := testConfigs.tiny

		network1 := createAndVerifyNetwork(t, config, "deterministic_test_1")
		network2 := createAndVerifyNetwork(t, config, "deterministic_test_2")

		for i := 0; i < network1.GetLayerCount(); i++ {
			layer1, _ := network1.Layer(i)
			layer2, _ := network2.Layer(i)

			for j := range layer1.Weights {
				copy(layer2.Weights[j], layer1.Weights[j])
			}

			copy(layer2.Biases, layer1.Biases)
		}

		tests := [][]float64{
			{1.0, 0.5, -0.5},
			{0.0, 0.0, 0.0},
			{-1.0, -2.0, -3.0},
			{10.0, 20.0, 30.0},
		}

		for _, input := range tests {
			output1, _ := network1.Forward(input)
			output2, _ := network2.Forward(input)

			if len(output1) != len(output2) {
				t.Fatalf("Output length mismatch: %d vs %d", len(output1), len(output2))
			}

			for i := range output1 {
				if math.Abs(output1[i]-output2[i]) > 1e-10 {
					t.Errorf("Non-deterministic behavior for input %v: output1[%d] = %f, output2[%d] = %f",
						input, i, output1[i], i, output2[i])
				}
			}
		}
	})
	t.Run("activation_flow", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name       string
			config     NetworkConfig
			verifyFunc func(output [][]float64) error
		}{
			{
				"all_relu_non_negative",
				NetworkConfig{
					LayerSizes:   []int{3, 4, 2, 1},
					Activations:  []string{"relu", "relu", "relu"},
					LearningRate: 0.01,
				},
				func(output [][]float64) error {
					for layer, output := range output {
						for i, val := range output {
							if val < 0 {
								return fmt.Errorf("Layer %d output %d is negative: %f", layer, i, val)
							}
						}
					}
					return nil
				},
			},
			{
				"sigmoid_bounded",
				NetworkConfig{
					LayerSizes:   []int{3, 3, 1},
					Activations:  []string{"sigmoid", "sigmoid"},
					LearningRate: 0.01,
				},
				func(output [][]float64) error {
					for layer, output := range output {
						for i, val := range output {
							if val < 0 || val > 1 {
								return fmt.Errorf("Layer %d output %d is out of bounds: %f", layer, i, val)
							}
						}
					}
					return nil
				},
			},
			{
				"tanh_bounded",
				NetworkConfig{
					LayerSizes:   []int{3, 2, 1},
					Activations:  []string{"tanh", "tanh"},
					LearningRate: 0.01,
				},
				func(output [][]float64) error {
					for layer, output := range output {
						for i, val := range output {
							if val < -1 || val > 1 {
								return fmt.Errorf("Layer %d output %d is out of bounds: %f", layer, i, val)
							}
						}
					}
					return nil
				},
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				network := createAndVerifyNetwork(t, tc.config, tc.name)

				testInputs := [][]float64{
					{1.0, -1.0, 1.0},
					{100.0, -100.0, 100.0},
					{0.0, 0.0, 0.0},
				}

				for _, input := range testInputs {
					intermediateResults, err := network.ForwardWithIntermediateResults(input)
					if err != nil {
						t.Fatalf("Forward(%v) returned error: %v", input, err)
					}
					if err := tc.verifyFunc(intermediateResults); err != nil {
						t.Errorf("Verification failed for input %v ", input)
					}
				}
			})
		}
	})
	t.Run("network_capacity", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name     string
			config   NetworkConfig
			capacity string
		}{
			{
				"underfitting_network",
				NetworkConfig{
					LayerSizes:   []int{10, 2, 1},
					Activations:  []string{"linear", "linear"},
					LearningRate: 0.01,
				},
				"low",
			},
			{
				"balanced_network",
				NetworkConfig{
					LayerSizes:   []int{10, 20, 10, 1},
					Activations:  []string{"relu", "relu", "sigmoid"},
					LearningRate: 0.01,
				},
				"medium",
			},
			{
				"overfitting_network",
				NetworkConfig{
					LayerSizes:   []int{10, 100, 100, 50, 1},
					Activations:  []string{"relu", "relu", "relu", "sigmoid"},
					LearningRate: 0.01,
				},
				"high",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				network := createAndVerifyNetwork(t, tc.config, tc.name)

				totalParams := 0
				for i := 0; i < network.GetLayerCount(); i++ {
					layer, _ := network.Layer(i)

					totalParams += layer.InputSize() * layer.OutputSize() // Weights
					totalParams += layer.OutputSize()                     // Biases
				}
				t.Logf("%s network has %d total parameters (capacity: %s)", tc.name, totalParams, tc.capacity)

				input := make([]float64, tc.config.LayerSizes[0])
				for i := range input {
					input[i] = 0.5 // Simple constant input for testing
				}

				output, err := network.Forward(input)
				if err != nil {
					t.Errorf("Forward pass failed for %s network: %v", tc.name, err)
				}

				if len(output) != tc.config.LayerSizes[len(tc.config.LayerSizes)-1] {
					t.Errorf("Output size mismatch for %s network: expected %d, got %d", tc.name, tc.config.LayerSizes[len(tc.config.LayerSizes)-1], len(output))
				}
			})
		}
	})
}

func TestNetworkIntegration(t *testing.T) {
	t.Run("end_to_end_pipeline", func(t *testing.T) {
		t.Parallel()
		t.Run("classification_pipeline", func(t *testing.T) {
			config := NetworkConfig{
				LayerSizes:   []int{4, 8, 4, 1},
				Activations:  []string{"relu", "relu", "sigmoid"},
				LearningRate: 0.01,
			}
			network := createAndVerifyNetwork(t, config, "classification_pipeline")

			tests := []struct {
				name        string
				input       []float64
				description string
			}{
				{"close_0_like", []float64{0.1, 0.2, 0.1, 0.2}, "low values"},
				{"class_1_like", []float64{0.9, 0.8, 0.9, 0.8}, "high values"},
				{"mix_features", []float64{0.1, 0.9, 0.1, 0.9}, "alternating features"},
				{"boundary_case", []float64{0.5, 0.5, 0.5, 0.5}, "all medium"},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					output, _ := network.Forward(tc.input)

					if output[0] < 0 || output[0] > 1 {
						t.Errorf("Classification output %v not in [0, 1] for %s", output[0], tc.description)
					}
					t.Logf("Input: %v, Output: %v (%s)", tc.input, output[0], tc.description)
				})
			}
		})
		t.Run("regression_pipeline", func(t *testing.T) {
			config := NetworkConfig{
				LayerSizes:   []int{3, 10, 5, 1},
				Activations:  []string{"tanh", "tanh", "linear"},
				LearningRate: 0.001,
			}
			network, _ := NewMLP(config)

			inputs := [][]float64{
				{-1.0, 0.0, 1.0},
				{0.0, 0.0, 0.0},
				{2.5, -1.5, 0.5},
				{-10.0, 10.0, 0.0},
			}

			outputs := make([]float64, len(inputs))
			for i, input := range inputs {
				output, _ := network.Forward(input)
				outputs[i] = output[0]

				if math.IsNaN(outputs[i]) || math.IsInf(outputs[i], 0) {
					t.Errorf("Regression output not finite: %v for input %v", outputs[0], input)
				}
			}

			allSame := true
			for i := 1; i < len(outputs); i++ {
				if outputs[i] != outputs[0] {
					allSame = false
					break
				}
			}

			if allSame {
				t.Errorf("All regression outputs are the same: %v", outputs)
			}
		})
		t.Run("multi_task_pipeline", func(t *testing.T) {
			config := NetworkConfig{
				LayerSizes:   []int{5, 10, 3},
				Activations:  []string{"relu", "sigmoid"},
				LearningRate: 0.01,
			}
			network := createAndVerifyNetwork(t, config, "multi_task_pipeline")

			input := []float64{0.2, 0.8, 0.5, 0.3, 0.9}
			output, _ := network.Forward(input)

			if len(output) != 3 {
				t.Errorf("Expected 3 outputs for multi-task, got %d", len(output))
			}

			sum := 0.0
			for i, val := range output {
				if val < 0 || val > 1 {
					t.Errorf("Output %d is out of bounds: %f", i, val)
				}
				sum += val
			}

			t.Logf("Multi-task output: %v, sum: %f", output, sum)
		})
	})
	t.Run("data_flow_integrity", func(t *testing.T) {
		t.Parallel()
		t.Run("batch_processing_simulation", func(t *testing.T) {
			config := NetworkConfig{
				LayerSizes:   []int{10, 20, 10, 2},
				Activations:  []string{"relu", "relu", "sigmoid"},
				LearningRate: 0.01,
			}
			network := createAndVerifyNetwork(t, config, "batch_processing")

			batchSize := 100
			results := make([][]float64, batchSize)

			for i := 0; i < batchSize; i++ {
				input := generateRandomInput(10)

				output, err := network.Forward(input)
				if err != nil {
					t.Errorf("Forward pass failed for batch input %d: %v", i, err)
					continue
				}
				results[i] = output
			}

			// Verify all results are valid
			validCount := 0
			for i, result := range results {
				if result != nil && len(result) == 2 {
					validCount++
					for j, val := range result {
						if math.IsNaN(val) || math.IsInf(val, 0) {
							t.Errorf("Invalid output at batch %d, index %d: %v", i, j, val)
						}
					}
				}
			}

			if validCount != batchSize {
				t.Errorf("Only %d/%d batch samples processed successfully", validCount, batchSize)
			}
		})
		t.Run("sequential_data_flow", func(t *testing.T) {
			config := NetworkConfig{
				LayerSizes:   []int{5, 3, 1},
				Activations:  []string{"tanh", "sigmoid"},
				LearningRate: 0.01,
			}
			network := createAndVerifyNetwork(t, config, "sequential_data_flow")

			// Process same input multiple times
			input := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
			outputs := make([]float64, 10)

			for i := 0; i < 10; i++ {
				output, err := network.Forward(input)
				if err != nil {
					t.Fatalf("Forward %d failed: %v", i, err)
				}
				outputs[i] = output[0]
			}

			// All outputs should be identical
			for i := 1; i < len(outputs); i++ {
				if outputs[i] != outputs[0] {
					t.Errorf("Output %d (%v) differs from output 0 (%v)", i, outputs[i], outputs[0])
				}
			}
		})
		t.Run("concurrent_forward_passes", func(t *testing.T) {
			config := NetworkConfig{
				LayerSizes:   []int{8, 16, 8, 4},
				Activations:  []string{"relu", "relu", "sigmoid"},
				LearningRate: 0.01,
			}
			network := createAndVerifyNetwork(t, config, "concurrent_forward")

			// Run concurrent forward passes
			numGoroutines := 10
			numIterations := 100
			var wg sync.WaitGroup
			errors := make(chan error, numGoroutines*numIterations)

			for g := 0; g < numGoroutines; g++ {
				wg.Add(1)
				go func(goroutineID int) {
					defer wg.Done()

					// Each goroutine gets unique input
					input := make([]float64, 8)
					for i := range input {
						input[i] = float64(goroutineID) * 0.1
					}

					for i := 0; i < numIterations; i++ {
						output, err := network.Forward(input)
						if err != nil {
							errors <- fmt.Errorf("goroutine %d iteration %d: %v", goroutineID, i, err)
							return
						}

						// Verify output
						if len(output) != 4 {
							errors <- fmt.Errorf("goroutine %d: output size %d, want 4", goroutineID, len(output))
						}
					}
				}(g)
			}

			wg.Wait()
			close(errors)

			// Check for errors
			errorCount := 0
			for err := range errors {
				t.Error(err)
				errorCount++
				if errorCount > 5 {
					t.Error("Too many errors, stopping error reporting")
					break
				}
			}
		})
	})
	t.Run("edge_case_handling", func(t *testing.T) {
		t.Parallel()

		t.Run("extreme_network_architectures", func(t *testing.T) {
			extremeConfigs := []struct {
				name   string
				config NetworkConfig
			}{
				{
					"very_deep",
					NetworkConfig{
						LayerSizes:   []int{10, 8, 6, 4, 3, 2, 1},
						Activations:  []string{"relu", "relu", "relu", "relu", "relu", "sigmoid"},
						LearningRate: 0.001,
					},
				},
				{
					"very_wide",
					NetworkConfig{
						LayerSizes:   []int{5, 100, 5},
						Activations:  []string{"relu", "sigmoid"},
						LearningRate: 0.001,
					},
				},
				{
					"bottleneck",
					NetworkConfig{
						LayerSizes:   []int{100, 10, 1, 10, 100},
						Activations:  []string{"relu", "linear", "relu", "sigmoid"},
						LearningRate: 0.001,
					},
				},
				{
					"expanding",
					NetworkConfig{
						LayerSizes:   []int{1, 10, 100, 1000},
						Activations:  []string{"relu", "relu", "linear"},
						LearningRate: 0.0001,
					},
				},
			}

			for _, tc := range extremeConfigs {
				t.Run(tc.name, func(t *testing.T) {
					if testing.Short() && tc.name == "expanding" {
						t.Skip("Skipping large network in short mode")
					}
					network := createAndVerifyNetwork(t, tc.config, tc.name)

					// Create appropriate input
					input := make([]float64, tc.config.LayerSizes[0])
					for i := range input {
						input[i] = rand.Float64()*0.2 - 0.1 // Small values to avoid overflow
					}

					output, err := network.Forward(input)
					if err != nil {
						t.Errorf("%s network forward failed: %v", tc.name, err)
					}

					expectedOutput := tc.config.LayerSizes[len(tc.config.LayerSizes)-1]
					if len(output) != expectedOutput {
						t.Errorf("%s network output size = %d; want %d", tc.name, len(output), expectedOutput)
					}
				})
			}
		})
		t.Run("numerical_edge_cases", func(t *testing.T) {
			network := createAndVerifyNetwork(t, testConfigs.tiny, "numerical_edge_cases")

			edgeCases := []struct {
				name  string
				input []float64
			}{
				{"very_small_positive", []float64{1e-10, 1e-10, 1e-10}},
				{"very_small_negative", []float64{-1e-10, -1e-10, -1e-10}},
				{"very_large_positive", []float64{1e10, 1e10, 1e10}},
				{"very_large_negative", []float64{-1e10, -1e10, -1e10}},
				{"mixed_extreme", []float64{1e10, -1e10, 0}},
				{"near_zero", []float64{1e-300, -1e-300, 0}},
			}

			for _, tc := range edgeCases {
				t.Run(tc.name, func(t *testing.T) {
					output, err := network.Forward(tc.input)
					if err != nil {
						t.Errorf("%s: forward failed: %v", tc.name, err)
						return
					}

					// Output should still be valid
					if math.IsNaN(output[0]) {
						t.Errorf("%s: output is NaN", tc.name)
					}

					// For sigmoid final layer, output should be in [0,1]
					if output[0] < 0 || output[0] > 1 {
						t.Errorf("%s: sigmoid output %v not in [0,1]", tc.name, output[0])
					}

					t.Logf("%s: input %v -> output %v", tc.name, tc.input, output[0])
				})
			}
		})
	})
	t.Run("performance_characteristics", func(t *testing.T) {
		t.Parallel()

		t.Run("forward_pass_scaling", func(t *testing.T) {
			if testing.Short() {
				t.Skip("Skipping performance test in short mode")
			}

			configs := []struct {
				name   string
				config NetworkConfig
			}{
				{"small", testConfigs.small},
				{"medium", testConfigs.medium},
				{"large", testConfigs.large},
			}

			for _, tc := range configs {
				t.Run(tc.name, func(t *testing.T) {
					network := createAndVerifyNetwork(t, tc.config, tc.name)

					input := make([]float64, tc.config.LayerSizes[0])
					for i := range input {
						input[i] = rand.Float64()
					}

					// Warm up
					for i := 0; i < 10; i++ {
						network.Forward(input)
					}

					// Time forward passes
					iterations := 1000
					start := time.Now()

					for i := 0; i < iterations; i++ {
						_, err := network.Forward(input)
						if err != nil {
							t.Fatalf("Forward pass failed: %v", err)
						}
					}

					duration := time.Since(start)
					avgTime := duration / time.Duration(iterations)

					t.Logf("%s network: %v per forward pass", tc.name, avgTime)
				})
			}
		})
	})
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================
func TestNetworkProperties(t *testing.T) {
	t.Run("mathematical_properties", func(t *testing.T) {
		t.Parallel()
		t.Run("output_dimension_invariant", func(t *testing.T) {
			// Property: Output size should always match last layer size
			configs := []NetworkConfig{
				{LayerSizes: []int{5, 3, 1}, Activations: []string{"relu", "sigmoid"}, LearningRate: 0.01},
				{LayerSizes: []int{10, 20, 5}, Activations: []string{"tanh", "linear"}, LearningRate: 0.01},
				{LayerSizes: []int{2, 4, 8, 16}, Activations: []string{"relu", "relu", "sigmoid"}, LearningRate: 0.01},
			}

			for i, config := range configs {
				configName := fmt.Sprintf("config_%d", i)
				network := createAndVerifyNetwork(t, config, configName)

				// Test with random inputs
				for j := 0; j < 10; j++ {
					input := generateRandomInput(config.LayerSizes[0])

					output, err := network.Forward(input)
					if err != nil {
						t.Errorf("Forward failed: %v", err)
						continue
					}

					expectedSize := config.LayerSizes[len(config.LayerSizes)-1]
					if len(output) != expectedSize {
						t.Errorf("Output size = %d; want %d", len(output), expectedSize)
					}
				}
			}
		})
		t.Run("activation_bounds", func(t *testing.T) {
			// Property: Activations should respect their mathematical bounds
			boundTests := []struct {
				name       string
				activation string
				minBound   float64
				maxBound   float64
			}{
				{"sigmoid_bounds", "sigmoid", 0, 1},
				{"tanh_bounds", "tanh", -1, 1},
				{"relu_bounds", "relu", 0, math.Inf(1)},
			}

			for _, tc := range boundTests {
				t.Run(tc.name, func(t *testing.T) {
					config := NetworkConfig{
						LayerSizes:   []int{5, 3, 1},
						Activations:  []string{"linear", tc.activation},
						LearningRate: 0.01,
					}
					network := createAndVerifyNetwork(t, config, tc.name)

					// Test with extreme inputs
					extremeInputs := [][]float64{
						{100, 100, 100, 100, 100},
						{-100, -100, -100, -100, -100},
						{-100, 100, -100, 100, -100},
						{0, 0, 0, 0, 0},
					}

					for _, input := range extremeInputs {
						output, _ := network.Forward(input)

						if tc.maxBound != math.Inf(1) && output[0] > tc.maxBound {
							t.Errorf("%s: output %v exceeds max bound %v", tc.name, output[0], tc.maxBound)
						}
						if output[0] < tc.minBound {
							t.Errorf("%s: output %v below min bound %v", tc.name, output[0], tc.minBound)
						}
					}
				})
			}
		})
		t.Run("scale_equivariance", func(t *testing.T) {
			// Property: For linear networks, scaling input should scale output
			config := NetworkConfig{
				LayerSizes:   []int{3, 2, 1},
				Activations:  []string{"linear", "linear"},
				LearningRate: 0.01,
			}
			network := createAndVerifyNetwork(t, config, "scale_equivariance")
			// set all biases to zeros
			for i := range network.Layers() {
				l, _ := network.Layer(i)
				for j := range l.Biases {
					l.Biases[j] = 0.0
				}
			}

			baseInput := []float64{1.0, 2.0, 3.0}
			baseOutput, _ := network.Forward(baseInput)

			scales := []float64{2.0, 0.5, -1.0, 10.0}

			for _, scale := range scales {
				scaledInput := make([]float64, len(baseInput))
				for i := range scaledInput {
					scaledInput[i] = baseInput[i] * scale
				}

				scaledOutput, _ := network.Forward(scaledInput)
				expectedOutput := baseOutput[0] * scale
				if math.Abs(scaledOutput[0]-expectedOutput) > 1e-10 {
					t.Errorf("Scale %v: output %v != expected %v", scale, scaledOutput[0], expectedOutput)
				}
			}
		})
		t.Run("non_negative_relu_propagation", func(t *testing.T) {
			// Property: All-ReLU network should maintain non-negativity
			config := NetworkConfig{
				LayerSizes:   []int{4, 3, 2, 1},
				Activations:  []string{"relu", "relu", "relu"},
				LearningRate: 0.01,
			}

			network := createAndVerifyNetwork(t, config, "non_negative_relu")

			// Set all weights and biases to positive values
			for i := 0; i < network.GetLayerCount(); i++ {
				layer, _ := network.Layer(i)
				for j := range layer.Weights {
					for k := range layer.Weights[j] {
						layer.Weights[j][k] = math.Abs(layer.Weights[j][k])
					}
				}
				for j := range layer.Biases {
					layer.Biases[j] = math.Abs(layer.Biases[j])
				}
			}

			// Test with positive inputs
			positiveInputs := [][]float64{
				{1, 2, 3, 4},
				{0.1, 0.2, 0.3, 0.4},
				{10, 20, 30, 40},
			}

			for _, input := range positiveInputs {
				output, _ := network.Forward(input)
				if output[0] < 0 {
					t.Errorf("All-positive ReLU network produced negative output: %v", output[0])
				}
			}
		})
		t.Run("zero_input_bias_isolation", func(t *testing.T) {
			// Property: With zero input, output depends only on biases
			config := NetworkConfig{
				LayerSizes:   []int{3, 2, 1},
				Activations:  []string{"linear", "linear"},
				LearningRate: 0.01,
			}

			network := createAndVerifyNetwork(t, config, "zero_input_bias")

			// Set weights to large values to ensure they would dominate if input wasn't zero
			for i := 0; i < network.GetLayerCount(); i++ {
				layer, _ := network.Layer(i)
				for j := range layer.Weights {
					for k := range layer.Weights[j] {
						layer.Weights[j][k] = 1000.0
					}
				}
			}

			zeroInput := []float64{0.0, 0.0, 0.0}
			output1, _ := network.Forward(zeroInput)

			// Change weights (shouldn't affect output with zero input)
			layer, _ := network.Layer(0)
			layer.Weights[0][0] = -1000.0

			output2, _ := network.Forward(zeroInput)

			if math.Abs(output1[0]-output2[0]) > 1e-10 {
				t.Errorf("Zero input property failed: weights affected zero-input output")
			}
		})
	})
	t.Run("robustness_properties", func(t *testing.T) {
		t.Parallel()
		t.Run("input_perturbuation_stability", func(t *testing.T) {
			// Small input changes should produce small output changes
			config := NetworkConfig{
				LayerSizes:   []int{5, 10, 5, 1},
				Activations:  []string{"tanh", "tanh", "sigmoid"},
				LearningRate: 0.01,
			}

			network := createAndVerifyNetwork(t, config, "perturbation_stability")

			baseInput := []float64{0.5, 0.5, 0.5, 0.5, 0.5}
			baseOutput, _ := network.Forward(baseInput)

			// Test small perturbations
			epsilon := 0.001
			for i := 0; i < 10; i++ {
				perturbedInput := make([]float64, len(baseInput))
				copy(perturbedInput, baseInput)

				// Add small random perturbation
				for j := range perturbedInput {
					perturbedInput[j] += (rand.Float64()*2 - 1) * epsilon
				}

				perturbedOutput, _ := network.Forward(perturbedInput)

				outputDiff := math.Abs(perturbedOutput[0] - baseOutput[0])
				if outputDiff > 0.1 { // Reasonable threshold for stability
					t.Logf("Large output change %v for small input perturbation", outputDiff)
				}
			}
		})
		t.Run("memory_stability", func(t *testing.T) {
			if testing.Short() {
				t.Skip("Skipping memory test in short mode")
			}

			network := createAndVerifyNetwork(t, testConfigs.medium, "memory_stability")
			input := generateRandomInput(testConfigs.medium.LayerSizes[0])

			// Get initial memory stats
			var m1, m2 runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&m1)

			// Perform many forward passes
			iterations := 10000
			for i := 0; i < iterations; i++ {
				_, err := network.Forward(input)
				if err != nil {
					t.Fatalf("Forward pass %d failed: %v", i, err)
				}

				if i%1000 == 0 {
					runtime.GC()
				}
			}

			// Get final memory stats
			runtime.GC()
			runtime.ReadMemStats(&m2)

			memGrowth := m2.Alloc - m1.Alloc
			if memGrowth > 10*1024*1024 { // 10MB threshold
				t.Logf("Significant memory growth detected: %d bytes", memGrowth)
			}
		})
		t.Run("weight_magnitude_robustness", func(t *testing.T) {
			// Property: Scaling all weights by same factor should scale output predictably
			config := NetworkConfig{
				LayerSizes:   []int{2, 3, 1},
				Activations:  []string{"linear", "linear"},
				LearningRate: 0.01,
			}

			network := createAndVerifyNetwork(t, config, "weight_magnitude_robustness")

			// Set biases to zero for clean scaling
			for i := 0; i < network.GetLayerCount(); i++ {
				layer, _ := network.Layer(i)
				for j := range layer.Biases {
					layer.Biases[j] = 0.0
				}
			}

			input := []float64{1.0, 1.0}
			originalOutput, _ := network.Forward(input)

			// Scale all weights by 2
			for i := 0; i < network.GetLayerCount(); i++ {
				layer, _ := network.Layer(i)
				for j := range layer.Weights {
					for k := range layer.Weights[j] {
						layer.Weights[j][k] *= 2.0
					}
				}
			}

			scaledOutput, _ := network.Forward(input)

			// For 2-layer linear network with zero bias, scaling all weights by 2
			// should scale output by 2^(number of layers) = 4
			expectedScale := math.Pow(2.0, float64(network.GetLayerCount()))
			expectedOutput := originalOutput[0] * expectedScale

			if math.Abs(scaledOutput[0]-expectedOutput) > 1e-10 {
				t.Errorf("Weight scaling property failed: got %v, want %v",
					scaledOutput[0], expectedOutput)
			}
		})
	})
	t.Run("numerical_properties", func(t *testing.T) {
		t.Parallel()

		t.Run("gradient_flow_readiness", func(t *testing.T) {
			// Property: Network should be ready for gradient computation
			// (This prepares for backpropagation implementation)
			config := NetworkConfig{
				LayerSizes:   []int{4, 6, 3, 1},
				Activations:  []string{"relu", "tanh", "sigmoid"},
				LearningRate: 0.01,
			}

			network := createAndVerifyNetwork(t, config, "gradient_flow_readiness")
			input := []float64{0.5, -0.3, 0.8, -0.1}

			// Get intermediate results for gradient computation later
			intermediates, err := network.ForwardWithIntermediateResults(input)
			if err != nil {
				t.Fatalf("Forward with intermediates failed: %v", err)
			}

			verifyIntermediateResults(t, network, input, intermediates)

			// Verify intermediate values are numerically stable
			for i, intermediate := range intermediates {
				for j, val := range intermediate {
					if math.IsNaN(val) || math.IsInf(val, 0) {
						t.Errorf("Intermediate layer %d value %d is not finite: %v", i, j, val)
					}
				}
			}
		})

		t.Run("numerical_precision_limits", func(t *testing.T) {
			// Property: Network should handle numerical precision limits gracefully
			network := createSimpleLinearNetwork(2, 1)
			layer, _ := network.Layer(0)

			// Test with very small weights
			setLayerWeightsAndBiases(layer,
				[][]float64{{1e-15, 1e-15}},
				[]float64{1e-15})

			input := []float64{1e15, 1e15}
			output, err := network.Forward(input)

			if err != nil {
				t.Errorf("Network failed with precision limit inputs: %v", err)
			}

			// Should produce finite result
			if math.IsNaN(output[0]) || math.IsInf(output[0], 0) {
				t.Errorf("Precision limit test produced non-finite result: %v", output[0])
			}
		})
	})
}

// ============================================================================
// BENCHMARKS
// ============================================================================

func BenchmarkMLPCreation(b *testing.B) {
	benchConfigs := []struct {
		name   string
		config NetworkConfig
	}{
		{"tiny", testConfigs.tiny},
		{"small", testConfigs.small},
		{"medium", testConfigs.medium},
		{"large", testConfigs.large},
		{"deep", testConfigs.deep},
		{"wide", testConfigs.wide},
	}
	for _, tc := range benchConfigs {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = NewMLP(tc.config)
			}
		})
	}
}

func BenchmarkMLPForward(b *testing.B) {
	benchConfigs := []struct {
		name   string
		config NetworkConfig
	}{
		{"tiny", testConfigs.tiny},
		{"small", testConfigs.small},
		{"medium", testConfigs.medium},
		{"large", testConfigs.large},
		{"deep", testConfigs.deep},
		{"wide", testConfigs.wide},
	}

	for _, tc := range benchConfigs {
		b.Run(tc.name, func(b *testing.B) {
			network, _ := NewMLP(tc.config)
			input := generateRandomInput(tc.config.LayerSizes[0])

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = network.Forward(input)
			}
		})
	}
}

func BenchmarkMLPForwardMemory(b *testing.B) {
	benchConfigs := []struct {
		name   string
		config NetworkConfig
	}{
		{"tiny", testConfigs.tiny},
		{"small", testConfigs.small},
		{"medium", testConfigs.medium},
		{"large", testConfigs.large},
		{"deep", testConfigs.deep},
		{"wide", testConfigs.wide},
	}

	for _, tc := range benchConfigs {
		b.Run(tc.name, func(b *testing.B) {
			network, _ := NewMLP(tc.config)
			input := generateRandomInput(tc.config.LayerSizes[0])

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = network.Forward(input)
			}
		})
	}
}

func BenchmarkMLPForwardActivation(b *testing.B) {
	activationNames := getActivationNames()

	for _, act := range activationNames {
		b.Run(act, func(b *testing.B) {
			config := NetworkConfig{
				LayerSizes:   []int{100, 50, 25, 10},
				Activations:  []string{act, act, act},
				LearningRate: 0.01,
			}

			network, _ := NewMLP(config)
			input := generateRandomInput(config.LayerSizes[0])

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = network.Forward(input)
			}
		})
	}
}

func BenchmarkMLPIntermediateResults(b *testing.B) {
	benchConfigs := []struct {
		name   string
		config NetworkConfig
	}{
		{"small", testConfigs.small},
		{"medium", testConfigs.medium},
		{"large", testConfigs.large},
		{"deep", testConfigs.deep},
	}

	for _, tc := range benchConfigs {
		b.Run(tc.name, func(b *testing.B) {
			network, _ := NewMLP(tc.config)
			input := generateRandomInput(tc.config.LayerSizes[0])

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = network.ForwardWithIntermediateResults(input)
			}
		})
	}
}

func BenchmarkMLPConcurrentForward(b *testing.B) {
	network, _ := NewMLP(testConfigs.medium)
	input := generateRandomInput(testConfigs.medium.LayerSizes[0])

	b.Run("sequential", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = network.Forward(input)
		}
	})

	b.Run("concurrent_2", func(b *testing.B) {
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				_, _ = network.Forward(input)
			}
		})
	})
}

func BenchmarkMLPBatchSimulation(b *testing.B) {
	network, _ := NewMLP(testConfigs.medium)

	batchSizes := []int{1, 10, 100, 1000}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("batch_%d", batchSize), func(b *testing.B) {
			// Pre-generate inputs
			inputs := make([][]float64, batchSize)
			for i := range inputs {
				inputs[i] = generateRandomInput(testConfigs.medium.LayerSizes[0])
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, input := range inputs {
					_, _ = network.Forward(input)
				}
			}
		})
	}
}
