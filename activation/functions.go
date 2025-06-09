package activation

import (
	"fmt"
	"math"
)

type ActivationFunc interface {
	Forward(x float64) float64
	String() string
}

type ReLU struct{}

func (r ReLU) Forward(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func (r ReLU) String() string {
	return "ReLU"
}

type Sigmoid struct{}

func (s Sigmoid) Forward(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (s Sigmoid) String() string {
	return "Sigmoid"
}

type Linear struct{}

func (l Linear) Forward(x float64) float64 {
	return x
}

func (l Linear) String() string {
	return "Linear"
}

type Tanh struct{}

func (t Tanh) Forward(x float64) float64 {
	return math.Tanh(x)
}

func (t Tanh) String() string {
	return "Tanh"
}

func NewActivation(name string) (ActivationFunc, error) {
	switch name {
	case "relu":
		return ReLU{}, nil
	case "sigmoid":
		return Sigmoid{}, nil
	case "linear":
		return Linear{}, nil
	case "tanh":
		return Tanh{}, nil
	case "":
		return nil, fmt.Errorf("activation function name cannot be empty")
	default:
		return nil, fmt.Errorf("unknown activation function: %q, supported: relu, sigmoid, linear, tanh", name)
	}
}
