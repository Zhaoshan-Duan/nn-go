package mlp

import (
	"neural-network-project/activation"
	"testing"
)

func TestNewMLP_InvalidLayerCount(t *testing.T) {
	_, err := NewMLP([]int{}, []string{}, 0.01)
	if err == nil {
		t.Error("expected error for empty layerSizes, got nil")
	}
}

func TestNewMLP_ActivationCountMismatch(t *testing.T) {
	_, err := NewMLP([]int{3, 2, 1}, []string{"relu"}, 0.01)
	if err == nil {
		t.Error("expected error for activation count mismatch, got nil")
	}
}

func TestNewMLP_NonPositiveLayerSize(t *testing.T) {
	_, err := NewMLP([]int{3, 0, 1}, []string{"relu", "sigmoid"}, 0.01)
	if err == nil {
		t.Error("expected error for non-positive layer size, got nil")
	}
}

func TestNetworkInitialization(t *testing.T) {
	tests := []struct {
		layerSizes   []int
		activations  []string
		learningRate float64
	}{
		{[]int{3, 2, 1}, []string{"relu", "sigmoid"}, 0.01},
		{[]int{784, 128, 64, 10}, []string{"relu", "relu", "sigmoid"}, 0.001},
		{[]int{4, 4, 4}, []string{"tanh", "tanh"}, 0.1},
		{[]int{2, 5}, []string{"linear"}, 0.05},
		{[]int{100, 50, 25, 1}, []string{"relu", "sigmoid", "linear"}, 0.02},
	}

	for _, test := range tests {
		mlp, err := NewMLP(test.layerSizes, test.activations, test.learningRate)
		if err != nil {
			t.Fatalf("Error creating MLP: %v", err)
		}

		if mlp.LearningRate != test.learningRate {
			t.Errorf("expected learning rate %v, got %v", test.learningRate, mlp.LearningRate)
		}

		if mlp.InputSize != test.layerSizes[0] {
			t.Errorf("expected input size %d, got %d", test.layerSizes[0], mlp.InputSize)
		}

		if mlp.OutputSize != test.layerSizes[len(test.layerSizes)-1] {
			t.Errorf("expected output size %d, got %d", test.layerSizes[len(test.layerSizes)-1], mlp.OutputSize)
		}

		expectedLayers := len(test.layerSizes) - 1
		if len(mlp.Layers) != expectedLayers {
			t.Errorf("expected %d layers, got %d", expectedLayers, len(mlp.Layers))
		}

		for i, layer := range mlp.Layers {
			expectedInputSize := test.layerSizes[i]
			expectedOutputSize := test.layerSizes[i+1]

			if len(layer.Weights[0]) != expectedInputSize {
				t.Errorf("layer %d: expected input size %d, got %d", i, expectedInputSize, len(layer.Weights[0]))
			}

			if len(layer.Weights) != expectedOutputSize {
				t.Errorf("layer %d: expected output size %d, got %d", i, expectedOutputSize, len(layer.Weights))
			}

			if len(layer.Biases) != expectedOutputSize {
				t.Errorf("layer %d: expected %d biases, got %d", i, expectedOutputSize, len(layer.Biases))
			}

			expectedActivation := test.activations[i]
			if layer.Activation.String() != activation.NewActivation(expectedActivation).String() {
				t.Errorf("layer %d: expected activation %s, got %s", i, expectedActivation, layer.Activation.String())
			}
		}
	}
}
