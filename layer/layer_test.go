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

func TestLayerForwardPass(t *testing.T) {
	testCases := []struct {
		inputSize  int
		outputSize int
		activation string
		input      []float64
	}{
		{3, 2, "relu", []float64{1.0, 2.0, 3.0}},
		{4, 4, "sigmoid", []float64{0.5, -1.2, 3.3, 0.0}},
		{2, 5, "tanh", []float64{2.2, -0.7}},
		{1, 1, "linear", []float64{42.0}},
	}

	for _, testCase := range testCases {
		act := activation.NewActivation(testCase.activation)
		layer := NewLayer(testCase.inputSize, testCase.outputSize, act)

		output := layer.Forward(testCase.input)

		if len(output) != testCase.outputSize {
			t.Errorf("expected output size %d, got %d", testCase.outputSize, len(output))
		}
	}
}

func TestLayerForwardKnownValues(t *testing.T) {
	layer := NewLayer(3, 1, activation.NewActivation("linear"))
	layer.Weights[0] = []float64{1.0, 2.0, 3.0}
	layer.Biases[0] = 4.0

	input := []float64{1.0, 1.0, 1.0}
	output := layer.Forward(input)
	expected := float64(1*1 + 2*1 + 3*1 + 4)

	if len(output) != 1 || output[0] != expected {
		t.Errorf("expected output [10], got %v", output)
	}
}
