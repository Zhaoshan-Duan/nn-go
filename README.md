# Multilayer Perceptron in Go

A multi-layer perceptron neural network in Go implemented from scratch, without external ML libraries.

## Features

- [x] Activation functions: ReLU, Sigmoid, Tanh, Linear
- [x] Layer struct with weights, biases, and activation
- [x] Layer forward pass
- [x] Network struct (MLP) with error handling and tests
- [ ] Forward propagation
- [ ] Backward propagation
- [ ] Training loop
- [ ] Data loading
- [ ] Evaluation/metrics
- [ ] CLI/Plotting

## Planned Subsystems

- Activation Function Subsystem
- Data Management Subsystem
- Network Architecture Subsystem
- Training Subsystem
- Model Persistence Subsystem
- Evaluation Subsystem

## Getting Started

### Running Tests

```sh
go test ./...
```

### Project Structure

```
neural-network-project/
├── activation/           # Activation functions (ReLU, Sigmoid, Tanh, etc.)
│   ├── activation.go
│   └── activation_test.go
├── layer/                # Layer struct and logic
│   ├── layer.go
│   └── layer_test.go
├── mlp/                  # Multi-layer perceptron (network struct)
│   ├── mlp.go
│   └── mlp_test.go
├── mathutil/             # Math utilities (if any)
│   ├── mathutil.go
│   └── mathutil_test.go
├── progress_tracker.md   # Implementation progress and notes
├── README.md             # Project overview and instructions
└── go.mod                # Go module file
```


- `activation/` – Implements activation functions 
- `layer/` – Contains layer struct, initialization, forward pass
- `mlp/` - Contains MLP (network) struct, constructor
- `main.go` – Entry point (to be implemented)

## Design Notes

This project uses a layer-first approach (rather than a neuron-first approach) for efficiency and simplicity, leveraging Go's strengths with slices and arrays.

For layers with fewer than 1000 neurons, the current sequential implementation is efficient and simple. For larger layers, I consider parallelizing the forward pass using goroutines to improve performance.

The network uses a simple constructor pattern for clarity and rapid prototyping. I consider reconfigure this into a builder pattern in the future. In the network, layers are stored as pointers with the MLP struct. This allows in-place updates and avoid unnecessary copying of large structs. 

## Roadmap

See `progress_tracker.md` for detailed progress.

## Implementation Plan

- **Phase 1:** Core implementation (activation functions, layer struct, forward propagation)
- **Phase 2:** Testing and refinement
- **Phase 3:** Extensions and optimizations

## Usage Examples (Placeholder)

_Examples will be added as components are implemented._
