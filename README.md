# Multilayer Perceptron in Go

A multi-layer perceptron neural network in Go implemented from scratch, without external ML libraries.

## Features

- [x] **Activation functions**: ReLU, Sigmoid, Tanh, Linear with comprehensive testing
- [x] **Math utilities**: DotProduct with zero-allocation performance
- [x] **Layer struct**: with weights, biases, and activation
- [x] Layer forward pass with thread-safe concurrent execution
- [x] Network struct (MLP) with comprehensive testing suite
- [x] Network Forward propagation with intermediate results for backpropagation
- [x] Data normalization: Z-score normalization with fit/transform pattern
- [x] Coffee roasting demo: Complete end-to-end ML pipeline validation
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

## Coffee Roasting Demo - Complete ML Pipeline
The project includes a complete end-to-end machine learning pipeline demonstrated with a coffee roasting classification problem, validating the implementation against lab examples from Coursera's Machine Learning Specialization.

```
✅ Data: 200 samples, Temperature 152-295°C, Duration 11.5-15.5min
✅ Normalization: [-1.7, 1.7] range (matches TensorFlow behavior)
✅ Network: 2→3→1 architecture with sigmoid activations
✅ Pretrained weights: Loaded from research paper
✅ Test Results:
   • Positive example (200°C, 13.9min): 1.0 probability → Good roast ✓
   • Negative example (200°C, 17min): 0.0 probability → Bad roast ✓
✅ Batch Performance: 80% accuracy on training samples
✅ Performance: 99.4ns per sample (10+ million predictions/second)

BenchmarkCoffeeRoastingPipeline-16         59277             19879 ns/op
```
### What this validates
- Mathematical correctness of forward propagation
- Proper activation function implementations
- Accurate weight loading and bias application
- Correct normalization behavior matching TensorFlow
- Production-ready inference performance


## Test Coverage Status
### Completed Test Suites
- **Activation Functions**: Unit, Property, Benchmark tests :heavy_check_mark:
- **Math Utilities**: Unit, Property, Benchmark tests with zero allocation :heavy_check_mark:
- **Layer**: Unit, Component, Integration, Property, Concurrency, Benchmark tests :heavy_check_mark:
- **Network (MLP)**: Unit, Component, Integration, Property, Concurrency, Benchmark tests :heavy_check_mark:
- **Data Normalization:** Unit, Property, Benchmark tests with mathematical guarantees :heavy_check_mark:
- **Coffee Demo Pipeline:** End-to-end integration tests with validation :heavy_check_mark:

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

# Run coffee demo test
go test -v -run TestCoffeeRoasting ./examples/coffee_demo

# Run coffee demo benchmarks
go test -bench=BenchmarkCoffeeRoasting ./examples/coffee_demo
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
│   ├── vector_test.go    # Comprehensive tests with benchmarks
│   ├── normalization.go  # Z-score normalization implementation
│   └── normalization_test.go # Property-based normalization tests
├── coffeedata/           # Coffee roasting dataset
│   └── data.go           # Synthetic coffee data generation
├── examples/
│   └── coffee_demo/
│       └── coffee_demo_test.go # Complete end-to-end ML pipeline test
├── progress_tracker.md   # Implementation progress and notes
├── README.md             # Project overview and instructions
└── go.mod                # Go module file
```

- `activation/` – Implements activation functions
- `mathutil/` - Math utilities with zero-allocation performance guarantees and normalization
- `layer/` – Contains layer struct, initialization, forward pass
- `network/` - Complete MLP network wtih configuraiton and pipeline support
- `coffeedata/` - Coffee roasting dataset generation and utilities
- `example/coffee_demo/` - End-to-end validation of ML pipeline with pretrained weights and biases


## Use Example
### Baisc Network Usage
```
package main

import (
    "fmt"
    "neural-network-project/network"
)

func main() {
    // Create a network configuration
    config := network.NetworkConfig{
        // Input -> Hidden1 -> Hidden2 -> Hidden3
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
### Inference with Normalization
```go
package main

import (
    "fmt"
    "neural-network-project/mathutil"
    "neural-network-project/network"
    "neural-network-project/coffeedata"
)

func main() {
    // Load data
    X, Y := coffeedata.LoadCoffeeData()

    // Normalize features
    normalizer := mathutil.NewNormalizer()
    XNormalized, err := normalizer.FitTransform(X)
    if err != nil {
        panic(err)
    }

    // Create and configure network
    config := network.NetworkConfig{
        LayerSizes:   []int{2, 3, 1},
        Activations:  []string{"sigmoid", "sigmoid"},
        LearningRate: 0.01,
    }

    mlp, err := network.NewMLP(config)
    if err != nil {
        panic(err)
    }

    // Set pretrained weights (from research paper)
    layer1, _ := mlp.Layer(0)
    layer1.Weights[0] = []float64{-8.93, -0.1}
    layer1.Weights[1] = []float64{0.29, -7.32}
    layer1.Weights[2] = []float64{12.9, 10.81}
    layer1.Biases = []float64{-9.82, -9.28, 0.96}

    layer2, _ := mlp.Layer(1)
    layer2.Weights[0] = []float64{-31.18, -27.59, -32.56}
    layer2.Biases[0] = 15.41

    // Make predictions
    for i := 0; i < len(XNormalized); i++ {
        prediction, err := mlp.Forward(XNormalized[i])
        if err != nil {
            panic(err)
        }

        decision := "Bad roast"
        if prediction[0] >= 0.5 {
            decision = "Good roast"
        }

        fmt.Printf("Sample %d: %.4f → %s (actual: %.0f)\n",
            i+1, prediction[0], decision, Y[i][0])
    }
}
```

## Design Notes

This project uses a layer-first approach (rather than a neuron-first approach) for efficiency and simplicity, leveraging Go's strengths with slices and arrays. My current implementation focuses on simplicity and correctness. Bencharmks show good performance for layers up to 1000 neurons. I consider implementing batch-level, and layer-level parallelism after profiling the network. Concurrency will be added based on profiling results, not speculation.

The network uses a simple constructor pattern for clarity and rapid prototyping. I consider reconfigure this into a builder pattern in the future. In the network, layers are stored as pointers with the MLP struct. This allows in-place updates and avoid unnecessary copying of large structs. The data normalization follows the standard ML fit/transform pattern, ensuring proper separation between training and inference phases.

## Clean Architecture Principles

The project follows clean package design principles to ensure maintainability, testability, and extensibility. Each package exposes only the necessary APIs while keeping implementation details private (except Layer's Weights and Biases since I need to manually set them to test before implementing backpropagation). All packages use constructor patterns with error handling to maintain stable interfaces. The dependency flow is unidirectional, with lower-level packages having no knowledge of their dependents.

### Testing Journey

When working on testing network forward prop, I realized that my initial testing strategy started to struggle with complex behaviors. It was getting confusing and unmanageable.

Up until `layer/`, I was doing plain unit test using simple table-driven testing. Individual unit tests are written before the implementation, and I only moved onto the next component if I had 100% coverage. But after some research, I learned that my tests have anti patterns. Moreover, the lack of clear testing levels to test the entire components and their subcomponents made me lack the confidence to proceed. There was a scaling issue.

Hence, in commits [6973ce6](https://github.com/Zhaoshan-Duan/nn-go/commit/6973ce6ecf840f2e7f22857406ac1e312473058c) (Testing Strategy Overhaul), I rewrote the tests for each package using a more structured, multi-level testing strategy with the following testing levels:

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
- **Phase 2.5:** Data normalization and example ML pipeline
- **Phase 3:** Backward propagation and training
- **Phase 4:** Extensions and optimizations (including targeted concurrency)
