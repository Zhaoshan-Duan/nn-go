package mlp

import (
	"fmt"
	"neural-network-project/activation"
	"neural-network-project/layer"
)

type MLP struct {
	Layers []*layer.Layer

	LearningRate float64
	InputSize    int
	OutputSize   int
}

func NewMLP(layerSizes []int, activations []string, learningRate float64) (*MLP, error) {
	if len(layerSizes) < 1 {
		return nil, fmt.Errorf("Network must have at least 1 layer.")
	}

	if len(activations) != len(layerSizes)-1 {
		return nil, fmt.Errorf("number of activations must be one less than number of size")
	}

	layers := make([]*layer.Layer, len(layerSizes)-1)

	for i := 0; i < len(layerSizes)-1; i++ {
		if layerSizes[i] <= 0 || layerSizes[i+1] <= 0 {
			return nil, fmt.Errorf("Layer sizes must be positive")
		}

		activation := activation.NewActivation(activations[i])
		layers[i] = layer.NewLayer(layerSizes[i], layerSizes[i+1], activation)
	}

	return &MLP{
		Layers:       layers,
		LearningRate: learningRate,
		InputSize:    layerSizes[0],
		OutputSize:   layerSizes[len(layerSizes)-1],
	}, nil
}

func Forward(input []float64) ([]float64, error) {
}
