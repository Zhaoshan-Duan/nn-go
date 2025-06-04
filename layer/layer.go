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
}

func NewLayer(inputSize, outputSize int, activation activation.ActivationFunc) *Layer {
	return &Layer{
		Weights:    initWeights(inputSize, outputSize),
		Biases:     initBiases(outputSize),
		Activation: activation,
	}
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

func (l *Layer) Forward(input []float64) []float64 {
	if len(input) != len(l.Weights[0]) {
		panic(fmt.Sprintf("Dimension Mismatch in Forward(): input length is %d, weight length is %d.", len(input), len(l.Weights)))
	}
	return l.forwardProp(input)
}

func (l *Layer) forwardProp(x []float64) []float64 {
	outputSize := len(l.Weights)
	outputs := make([]float64, outputSize)

	for i := range l.Weights {
		w_i := l.Weights[i]
		preActivation := mathutil.DotProduct(w_i, x) + l.Biases[i]
		z := l.Activation.Forward(preActivation)
		outputs[i] = z
	}
	return outputs
}
