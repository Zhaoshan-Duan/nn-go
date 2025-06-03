package layer

import (
	"neural-network-project/activation"
	"testing"
)

func TestNewLayerInitialization(t *testing.T) {
	tests := []struct {
		inputSize, outputSize int
		activation            string
	}{
		{3, 2, "relu"},
		{3, 1, "tanh"},
		{100, 100, "sigmoid"},
		{100, 100, "relu"},
		{10, 128, "linear"},
	}

	for _, test := range tests {
		actualLayer := NewLayer(test.inputSize, test.outputSize, activation.NewActivation(test.activation))

		if len(actualLayer.Weights) != test.outputSize {
			t.Errorf("expected %d rows in weights, got %d", test.outputSize, len(actualLayer.Weights))
		}

		for i := range actualLayer.Weights {
			if len(actualLayer.Weights[i]) != test.inputSize {
				t.Errorf("expected %d columns in weights[%d], got %d", test.inputSize, i, len(actualLayer.Weights[i]))
			}
		}

		if len(actualLayer.Biases) != test.outputSize {
			t.Errorf("expected %d biases, got %d", test.outputSize, len(actualLayer.Biases))
		}

		if actualLayer.Activation == nil {
			t.Error("activation function should not be nil")
		}
	}
}

func TestLayerWeightsBiasRange(t *testing.T) {
	layer := NewLayer(5, 5, activation.NewActivation("relu"))

	for i := range layer.Weights {
		for j := range layer.Weights[i] {
			if layer.Weights[i][j] < -1 || layer.Weights[i][j] > 1 {
				t.Errorf("weight out of range: %v", layer.Weights[i][j])
			}
		}
	}
	for _, b := range layer.Biases {
		if b < -1 || b > 1 {
			t.Errorf("bias out of range: %v", b)
		}
	}
}
