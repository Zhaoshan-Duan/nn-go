# Multilayer Perceptron in Go

A multi-layer perceptron neural network in Go implemented from scratch, without external ML libraries.

## Features

- [x] Activation functions: ReLU, Sigmoid, Tanh, Linear
- [x] Layer struct with weights, biases, and activation
- [ ] Layer forward pass
- [ ] Network struct (MLP)
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

- `activation/` – Activation functions and tests
- `layer/` – Layer struct and logic 
- `main.go` – Entry point (to be implemented)

## Design Notes

This project uses a layer-first approach (rather than a neuron-first approach) for efficiency and simplicity, leveraging Go's strengths with slices and arrays.

## Roadmap

See `progress_tracker.md` for detailed progress.

## Implementation Plan

- **Phase 1:** Core implementation (activation functions, layer struct, forward propagation)
- **Phase 2:** Testing and refinement
- **Phase 3:** Extensions and optimizations

## Usage Examples (Placeholder)

_Examples will be added as components are implemented._
