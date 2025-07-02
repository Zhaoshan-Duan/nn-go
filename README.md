# Multilayer Perceptron in Go

A multi-layer perceptron neural network in Go implemented from scratch, without external ML libraries.

## Features

- [x] **Activation functions**: ReLU, Sigmoid, Tanh, Linear with comprehensive testing
- [x] **Math utilities**: DotProduct with zero-allocation performance
- [x] **Layer struct**: with weights, biases, and activation
- [x] Layer forward pass with thread-safe concurrent execution
- [x] Network struct (MLP) with comprehensive testing suite
- [x] Network Forward propagation with intermediate results for backpropagation
- [ ] Backward propagation
- [ ] Training loop
- [ ] Data loading
- [ ] Evaluation/metrics
- [ ] CLI/Plotting

### Performance Highlights
- **DotProduct**: Linear scaling O(n) from 0.89ns (2 elements) to 2284ns (10,000 elements)
- **Zero allocation**: All core math operations verified to have zero memory allocation
- **Layer Forward Pass**:
  - Small layers (10×5): ~55-125 ns/op
  - Medium layers (100×50): ~2.3-3.6 μs/op
  - Large layers (784×128): ~64-69 μs/op
  - Only 1 memory allocation per forward pass
  - Thread-safe for concurrent inference

- **Network Forward Pass**:
  - Only **632B and 3 allocations** per forward pass for medium networks

| Network Size | Forward Pass Time | Memory/Op | Allocations |
|---| :---:|:---:|:---:|
| Tiny (3→2→1) | ~68ns | 24B | 2 |
| Small (10→5→1) | ~95ns | 56B | 2 |
| Medium (100→50→25→10) | ~3.7μs | 704B | 3 |
| Large (784→128→64→10) | ~72μs | 1616B | 3 |
| Deep (8 layers) | ~6.4μs | 2,408B | 8 |

- **Activation Function Performance**
  - Linear: Baseline (fastest)
  - ReLU: ~15% slower than linear
  - Sigmoid: ~54% slower than linear
  - Tanh: ~62% slower than linear

Concurrency Benefits
- Sequential processing: ~3.7μs/op for medium networks
- Concurrent (2 goroutines): ~566ns/op (6.5x speedup potential)

- **Mathematical Correctness**
  - **Property testing**: Mathematical invariants verified (commutativity, distributivity, etc.)
  - **Numerical stability**: Handles extreme values (infinity, NaN, very large/small numbers)
  - **Gradient readiness**: `ForwardWithIntermediateResults()` provides foundation for backpropagation

## Test Coverage Status
### Completed Test Suites
- **Activation Functions**: Unit, Property, Benchmark tests :heavy_check_mark:
- **Math Utilities**: Unit, Property, Benchmark tests with zero allocation :heavy_check_mark:
- **Layer**: Unit, Component, Integration, Property, Concurrency, Benchmark tests :heavy_check_mark:
- **Network (MLP)**: Unit, Component, Integration, Property, Concurrency, Benchmark tests :heavy_check_mark:

## Getting Started

### Running Tests

```sh
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run specific package tests
go test ./activation
go test ./mathutil
go test ./layer
go test ./network

# Run benchmarks
go test -bench=. ./..

# Run specific package benchmarks
go test -bench=. ./activation
go test -bench=. ./mathutil
go test -bench=. ./layer
go test -bench=. ./network

# Run benchmarks with memory allocation stats
go test -bench=. -benchmem ./activation
go test -bench=. -benchmem ./mathutil
go test -bench=. -benchmem ./layer
go test -bench=. -benchmem ./network

# Run with race detector to verify thread safety
go test -race ./..

# Run tests with coverage
go test -cover ./...
```

### Project Structure

```
neural-network-project/
├── activation/           # Activation functions (ReLU, Sigmoid, Tanh, etc.)
│   ├── functions.go
│   └── functions_test.go # Comprehensive tests with benchmarks
├── layer/                # Layer struct and logic
│   ├── layer.go
│   └── layer_test.go     # Comprehensive tests including concurrency
├── network/              # Multi-layer perceptron (network struct)
│   ├── mlp.go
│   └── mlp_test.go       # Comprehensive tests with intergration pipelines
├── mathutil/             # Math utilities
│   ├── vector.go
│   └── vector_test.go  # Comprehensive tests with benchmarks
├── progress_tracker.md   # Implementation progress and notes
├── README.md             # Project overview and instructions
└── go.mod                # Go module file
```

- `activation/` – Implements activation functions
- `mathutil/` - Math utilities with zero-allocation performance guarantees
- `layer/` – Contains layer struct, initialization, forward pass
- `network/` - Complete MLP network wtih configuraiton and pipeline support
- `main.go` – Entry point (to be implemented)

## Use Example
```
package main

import (
    "fmt"
    "neural-network-project/network"
)

func main() {
    // Create a network configuration
    config := network.NetworkConfig{
        LayerSizes:   []int{784, 128, 64, 10},
        Activations:  []string{"relu", "relu", "sigmoid"},
        LearningRate: 0.001,
    }

    // Create the network
    mlp, err := network.NewMLP(config)
    if err != nil {
        panic(err)
    }

    // Prepare input (e.g., flattened 28x28 image)
    input := make([]float64, 784)
    // ... fill input with data ...

    // Forward pass
    output, err := mlp.Forward(input)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Network output: %v\n", output)
}
```

## Design Notes

This project uses a layer-first approach (rather than a neuron-first approach) for efficiency and simplicity, leveraging Go's strengths with slices and arrays. My current implementation focuses on simplicity and correctness. Bencharmks show good performance for layers up to 1000 neurons. I consider implementing batch-level, and layer-level parallelism after profiling the network. Concurrency will be added based on profiling results, not speculation.

The network uses a simple constructor pattern for clarity and rapid prototyping. I consider reconfigure this into a builder pattern in the future. In the network, layers are stored as pointers with the MLP struct. This allows in-place updates and avoid unnecessary copying of large structs.

### Testing Journey

When working on testing network forward prop, I realized that my initial testing strategy started to struggle with complex behaviors. It was getting confusing and unmanageable.

Up until `layer/`, I was doing plain unit test using simple table-driven testing. Individual unit tests are written before the implementation, and I only moved onto the next component if I had 100% coverage. But after some research, I learned that my tests have anti patterns. Moreover, the lack of clear testing levels to test the entire components and their subcomponents made me lack the confidence to proceed. There was a scaling issue.

Hence, in commits [6973ce6](https://github.com/Zhaoshan-Duan/nn-go/commit/6973ce6ecf840f2e7f22857406ac1e312473058c) (Testing Strategy Overhaul), I rewrote the tests for each package. I rewrote the tests for each package using a more structured approach, multi-level testing strategy with the following testing levels:

1. **Unit Level** - Individual functions with known input/output pairs
2. **Component Level** - Whole components with realistic scenarios
3. **Integration Level** - Multiple components working together
4. **Property Level** - Mathematical invariants and properties (monotonicity, range, symmetry)
5. **Performance Level** - Benchmarks for speed and memory allocation
6. **Concurrency Level** - Thread-safety verification (added for layer package)

Tests are re-organized by sub-tests in parallel execution. Benchmarking was also added just to ensure I had zero memory allocation for activation functions, and I knew I wanted to incorporate concurrency for matrix operations later.

When adapting this apporach for `network/`, one single test script has become barely managable as it had 2880 lines (It was barely manageable for `layer/` already, which had 1767 lines). So at commit placeholder, I broken down the test script into individual sctips for each testing level.

## Roadmap

See `progress_tracker.md` for detailed progress and technical notes.

## Implementation Plan

- **Phase 1:** Core implementation (activation functions, layer struct, forward propagation)
- **Phase 2:** Network-level forward propagation and testing
- **Phase 3:** Backward propagation and training
- **Phase 4:** Extensions and optimizations (including targeted concurrency)
