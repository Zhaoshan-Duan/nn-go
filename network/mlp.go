package network

import (
	"fmt"
	"math"
	"neural-network-project/activation"
	"neural-network-project/layer"
	"strings"
)

// NetworkConfig represents the configuration for a Multi-Layer Perceptron (MLP) network.
type NetworkConfig struct {
	LayerSizes   []int    // Layer sizes for each layer in the network [input_size, hidden1, hidden2, ..., output_size]
	Activations  []string // Activation functions for each layer, must be one less than LayerSizes
	LearningRate float64
}

// MLP represents a Multi-Layer Perceptron neural network.
type MLP struct {
	layers       []*layer.Layer
	learningRate float64
	inputSize    int
	outputSize   int
}

// NewNetwork creates a new Multi-Layer Perceptron (MLP) with the given configuration.
func NewMLP(config NetworkConfig) (*MLP, error) {
	// Validate the network configuration
	if err := validateNetworkConfig(config); err != nil {
		return nil, err
	}

	layerSizes := config.LayerSizes
	activations := config.Activations

	// Create layers based on the provided sizes and activations
	layers := make([]*layer.Layer, len(layerSizes)-1)

	for i := 0; i < len(layerSizes)-1; i++ {
		// Create a new layer with the specified size and activation function
		activation, err := activation.NewActivation(activations[i])
		if err != nil {
			return nil, fmt.Errorf("invalid activation function: %s", activations[i])
		}

		layer, err := layer.NewLayer(layerSizes[i], layerSizes[i+1], activation)
		if err != nil {
			return nil, fmt.Errorf("failed to create layer %d: %v", i, err)
		}
		layers[i] = layer
	}

	return &MLP{
		layers:       layers,
		learningRate: config.LearningRate,
		inputSize:    layerSizes[0],
		outputSize:   layerSizes[len(layerSizes)-1],
	}, nil
}

func validateNetworkConfig(config NetworkConfig) error {
	if len(config.LayerSizes) == 1 {
		return fmt.Errorf("network must have at least 1 trainable layer")
	}

	if len(config.LayerSizes) < 2 {
		return fmt.Errorf("network must have at least 2 layers (input and output)")
	}

	for i, size := range config.LayerSizes {
		if size <= 0 {
			return fmt.Errorf("layer size must be positive, got %d at index %d", size, i)
		}
	}

	expectedActivations := len(config.LayerSizes) - 1
	if len(config.Activations) != expectedActivations {
		return fmt.Errorf("number of activations must be one less than layer sizes: expected %d, got %d", expectedActivations, len(config.Activations))
	}

	if config.LearningRate <= 0 || math.IsInf(config.LearningRate, 0) || math.IsNaN(config.LearningRate) {
		return fmt.Errorf("learning rate must be positive, got %f", config.LearningRate)
	}

	return nil
}

func (m *MLP) Forward(input []float64) ([]float64, error) {
	if input == nil {
		return nil, fmt.Errorf("input cannot be nil")
	}

	// Check if input size matches the network's input size
	if len(input) != m.InputSize() {
		return nil, fmt.Errorf("input size mismatch: network input size %d, input vector size %d", m.InputSize(), len(input))
	}

	// Forward propagate through each layer
	output, err := m.forwardProp(input)
	if err != nil {
		return nil, fmt.Errorf("Forward propagation error: %v", err)
	}

	return output, nil
}

func (m *MLP) ForwardWithIntermediateResults(input []float64) ([][]float64, error) {
	if input == nil {
		return nil, fmt.Errorf("input cannot be nil")
	}

	if len(input) != m.InputSize() {
		return nil, fmt.Errorf("input size mismatch: network input size %d, input vector size %d", m.InputSize(), len(input))
	}

	results := make([][]float64, 0, len(m.layers)+1)

	for i, l := range m.layers {
		output, err := l.Forward(input)
		if err != nil {
			return nil, fmt.Errorf("error in layer %d: %v", i, err)
		}
		results = append(results, output)
		input = output
	}
	return results, nil
}

func (m *MLP) forwardProp(input []float64) ([]float64, error) {
	currentOutput := input
	for i, layer := range m.layers {
		var err error
		currentOutput, err = layer.Forward(currentOutput)
		if err != nil {
			return nil, fmt.Errorf("error in layer %d: %v", i, err)
		}
	}

	return currentOutput, nil
}

func (m *MLP) GetLayerCount() int {
	return len(m.layers)
}

func (m *MLP) LearningRate() float64 {
	return m.learningRate
}

func (m *MLP) InputSize() int {
	return m.inputSize
}

func (m *MLP) OutputSize() int {
	return m.outputSize
}

func (m *MLP) Layers() []*layer.Layer {
	return m.layers
}

// Layer returns the layer at the specified index.
func (m *MLP) Layer(index int) (*layer.Layer, error) {
	if index < 0 || index >= len(m.layers) {
		return nil, fmt.Errorf("Layer index out of bounds: %d", index)
	}
	return m.layers[index], nil
}

func (m *MLP) String() string {
	var sb strings.Builder

	layerSizes := make([]string, len(m.layers)+1)
	layerSizes[0] = fmt.Sprintf("Input Layer: %d neurons", m.inputSize)
	for i, l := range m.layers {
		layerSizes[i+1] = fmt.Sprintf("Layer %d: %d neurons, Activation: %s", i+1, l.OutputSize(), l.Activation.String())
	}

	activations := make([]string, len(m.layers))
	for i, l := range m.layers {
		activations[i] = l.Activation.String()
	}

	sb.WriteString(fmt.Sprintf("MLP(layers=[%s], ", strings.Join(layerSizes, "→")))
	sb.WriteString(fmt.Sprintf("activations=[%s], ", strings.Join(activations, ", ")))
	sb.WriteString(fmt.Sprintf("learning_rate=%.4f)", m.learningRate))

	return sb.String()
}
