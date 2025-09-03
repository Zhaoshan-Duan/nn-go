# Multilayer Perceptron in Go

A multi-layer perceptron neural network in Go implemented from scratch, without external ML libraries.

## Current Implementation Status

### :white_check_mark: Completed Components
- **Activation functions**: ReLU, Sigmoid, Tanh, Linear with comprehensive testing
- **Math utilities**: DotProduct with zero-allocation performance; Z-score normalization with fit/transform pattern
- **Layer Implementation**: weights/biases storage and initialization; forward propagation with activation; thread-safe for concurrent reads
- **Network (MLP)**: configurable architecture via `NetworkConfig`; sequantial forward propagation; intermediate results for backpropagation
- **Example Pipeline**: Coffee roasting classification with pretrained weights and biases

### :construction: Not Yet Implemented
- [ ] Loss functions
- [ ] Backward propagation
- [ ] Training loop with gradient descent
- [ ] Model persistence
- [ ] Evaluation/metrics
- [ ] CLI/Plotting

## Performance Characteristics
### Activation Functions
As shown [here](benchmarks/activation_2025-07-17.txt), all activation functions achieve **zero memory allocation** per operation

| Activation      | Forward Pass | Batch Processing (1000 items) |
| :---        |    :----   | :--- |
| Linear     | 1.139ns      | 1.346μs                      |
| ReLU       | 1.139ns      | 1.440μs                      |
| Sigmoid    | 6.764ns      | 7.030μs                      |
| Tanh       | 9.284ns      | 9.805μs                      |

### Vector Operations
As shown in [here](benchmarks/mathutil_vector_2025-07-18.txt), all vector operations achieve **zero memory allocation** per operation with linear time complexity.

| Operation       | Tiny (2 elements) | Small (10) | Medium (100) | Large (1K) | Very Large (10K) |
|:---            |:---              |:---        |:---          |:---        |:---              |
| DotProduct     | 1.452ns           | 4.663ns     | 42.22ns       | 416.3ns      | 4.127μs           |
| Memory         | 0 B/op           | 0 B/op     | 0 B/op       | 0 B/op     | 0 B/op           |

### Data Normalization
As shown in [here](benchmarks/normalization-07-18.txt), Z-score normalization with linear time complexity $O(n×m)$ ($n$=samples, $m$=features).

**Sample Scaling** with 10 features fixed:

| Operation | 100 samples | 1000 samples | 10,000 samples |
| :--- |:---          |:---        |:---              |
| Fit |  1.453 μs | 16.09 μs | 228.9 μs |
| Transform |  4.876 μs | 56.37 μs | 778.6 μs |


**Feature Scaling** with 1,000 samples fixed:

| Operation | 10 features | 100 features | 10,000 features |
| :--- |:---          |:---        |:---              |
| Fit |  17.18 μs | 174.1 μs | 129.5 ms |
| Transform |  62.15 μs | 490.5 μs | 3.098 ms |

### Layer Forward Pass
As shown [here](benchmarks/layer_2025-07-23.txt), layer forward propagation achieves **single memory allocation** per operation with linear time complexity.

| Layer Size      | ReLU    | Sigmoid | Tanh    | Linear  | Memory    |
|:---            |:---     |:---     |:---     |:---     |:---       |
| Small (10→5)   | 62.63ns   | 137.5ns   | 124.0ns   | 65.44ns    | 1 allocs/op (48 B/op)  |
| Medium (100→50)| 3.153μs   | 3.970μs   | 3.987μs   | 2.587μs   | 1 alloc/op (416 B/op) |
| Large (784→128)| 74.13μs    | 73.81μs    | 80.29μs    | 65.52μs    |  1 allocs/op (1024 B/op) |
| Very Large (2048→1024)| 1.630ms| 1.641ms   | 1.841ms   | 1.861ms   |   1 allocs/op (8192 B/op) |

## Quick Start

```sh
# Clone the repository
git clone https://github.com/Zhaoshan-Duan/neural-network-project
cd neural-network-project

# Run tests
go test ./...

# Run with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./...

# Run the example
go run main.go
```

### Usage Example
#### Basic Inference
```go
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

#### Inference with Normalization

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

#### Coffee Roasting Demo
This project includes a complete example demonstration binary classification for coffee roasting quality based on Coursera's ML Specialization course labs.

```sh
// Temperature and duration determine roast quality
// Good roast: 175°C < temp < 260°C, 12min < duration < 15min
// with additional linear constraint

go test -v -run TestCoffeeRoasting ./demo
```

Results with pretrained weights:
- 200°C, 13.9min → 99.97% probability (Good roast ✓)
- 200°C, 17.0min → 0.02% probability (Bad roast ✓)

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

## Project Structure
```sh
neural-network-project/
├── activation/           # Activation functions
│   ├── functions.go
│   └── functions_test.go # tests with benchmarks
├── benchmarks/           # Performance benchmark for each componenet
│   ├── activation_2025-07-17.txt
│   ├── mathutil_vector_2025-07-18.txt
│   ├── normalization-07-20.txt
│   └── layer_2025-07-23.txt
├── mathutil/
│   ├── vector.go
│   ├── vector_test.go    # tests with benchmarks
│   ├── normalization.go  # Z-score normalization implementation
│   └── normalization_test.go # Property-based normalization tests
├── layer/
│   ├── layer.go
│   └── layer_test.go     # tests including concurrency
├── network/
│   ├── mlp.go
│   └── mlp_test.go       # tests with intergration pipelines
├── coffeedata/           # Coffee roasting dataset
│   └── data.go           # Synthetic coffee data generation
├── examples/
│   └── coffee_demo/
|       ├── main.go               # Example usage
│       └── coffee_demo_test.go # Complete end-to-end ML pipeline test
├── progress_tracker.md   # Implementation progress and notes
├── README.md             # Project overview and instructions
└── go.mod                # Go module file
```

## Testing Philosophy
This project follows a structured, multi-level testing apporach:
1. **Unit Tests** - Individual functions with known input/output pairs
2. **Component Tests** - Whole components with realistic scenarios
3. **Integration Tests** - Multiple components working together
4. **Property Tests** - Mathematical invariants (monotonicity, bounds, symmetry)
5. **Robustness Tests** - Edge cases, error recovery, extreme inputs, thread-safety
6. **Benchmarks** - Performance and memory allocation verification

### Test Organization
Each package follows a consistent test structure:
```sh
// TEST HELPERS
// UNIT TESTS (by function/method)
// COMPONENT TESTS (if applicable)
// INTEGRATION TESTS (if applicable)
// PROPERTY TESTS
// ROBUSTNESS TESTS
// BENCHMARKS
```

## Roadmap
See `progress_tracker.md` for detailed progress and technical notes.
