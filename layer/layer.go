package layer

import (
	"fmt"
	"math/rand/v2"
	"neural-network-project/activation"
	"neural-network-project/mathutil"
)

// Layer represents a fully-connected layer in the network
type Layer struct {
	// Weights is a 2D slice of slices
	// - 1st dimension is the number of neurons in this layer
	// - 2nd dimension is the number of inputs to each neuron
	Weights [][]float64

	// Biases is a 1D slice whose length is the number of neurons
	Biases []float64

	// Activation function to apply to the layer outputs
	Activation activation.ActivationFunc

	// Store dimensions for validation
	inputSize, outputSize int
}

func NewLayer(inputSize, outputSize int, activationFunc activation.ActivationFunc) (*Layer, error) {
	if inputSize <= 0 {
		return nil, fmt.Errorf("input size must be positive, got %d", inputSize)
	}

	if outputSize <= 0 {
		return nil, fmt.Errorf("output size must be positive, got %d", outputSize)
	}

	if activationFunc == nil {
		return nil, fmt.Errorf("activation function cannot be nil")
	}

	return &Layer{
		Weights:    initWeights(inputSize, outputSize),
		Biases:     initBiases(outputSize),
		Activation: activationFunc,
		inputSize:  inputSize,
		outputSize: outputSize,
	}, nil
}

func initWeights(inputSize, outputSize int) [][]float64 {
	weights := make([][]float64, outputSize)

	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			// Initialize weights with small random values
			weights[i][j] = rand.Float64()*2 - 1 // Random value between -1 and 1
		}
	}
	return weights
}

func initBiases(outputSize int) []float64 {
	biases := make([]float64, outputSize)

	for i := range biases {
		biases[i] = rand.Float64()*2 - 1
	}
	return biases
}

func (l *Layer) Forward(input []float64) ([]float64, error) {
	if input == nil {
		return nil, fmt.Errorf("input cannot be nil")
	}

	if len(input) == 0 {
		return nil, fmt.Errorf("input cannot be empty")
	}

	if len(input) != l.inputSize {
		return nil, fmt.Errorf("input size mismatch: expected %d got %d", l.inputSize, len(input))
	}

	if len(l.Weights) != l.outputSize {
		return nil, fmt.Errorf("layer weights output size mismatch: expected %d got %d", l.outputSize, len(l.Weights))
	}

	if l.Weights == nil {
		return nil, fmt.Errorf("layer weights cannot be nil")
	}

	if l.Biases == nil {
		return nil, fmt.Errorf("layer biases cannot be nil")
	}

	if len(l.Biases) != l.outputSize {
		return nil, fmt.Errorf("layer biases output size mismatch: expected %d got %d", l.outputSize, len(l.Biases))
	}

	for i, weightRow := range l.Weights {
		if len(weightRow) != l.inputSize {
			return nil, fmt.Errorf("neuron %d has %d weights, expected %d", i, len(weightRow), l.inputSize)
		}
	}

	if l.Activation == nil {
		return nil, fmt.Errorf("activation function cannot be nil")
	}

	return l.forwardProp(input), nil
}

func (l *Layer) forwardProp(x []float64) []float64 {
	outputs := make([]float64, l.outputSize)

	for i := range l.Weights {
		w_i := l.Weights[i]
		preActivation := mathutil.DotProduct(w_i, x) + l.Biases[i]
		z := l.Activation.Forward(preActivation)
		outputs[i] = z
	}
	return outputs
}

func (l *Layer) InputSize() int {
	return l.inputSize
}

func (l *Layer) OutputSize() int {
	return l.outputSize
}

func (l *Layer) String() string {
	return fmt.Sprintf("Layer(input=%d, output=%d, activation=%s)", l.inputSize, l.outputSize, l.Activation.String())
}
