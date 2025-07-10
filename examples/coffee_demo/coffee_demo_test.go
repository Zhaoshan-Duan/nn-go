package demo

import (
	"math"
	"neural-network-project/coffeedata"
	"neural-network-project/mathutil"
	"neural-network-project/network"
	"testing"
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func findMinMax(data [][]float64, col int) (float64, float64) {
	if len(data) == 0 {
		return 0, 0
	}
	min, max := data[0][col], data[0][col]

	for i := 1; i < len(data); i++ {
		if data[i][col] < min {
			min = data[i][col]
		}

		if data[i][col] > max {
			max = data[i][col]
		}

	}
	return min, max
}

func TestCoffeeRoasting(t *testing.T) {
	t.Run("complete_pipeline_with_pretrained_weights_and_biases", func(t *testing.T) {
		// Load coffee data
		X, Y := coffeedata.LoadCoffeeData()
		t.Logf("Loaded data: X shape (%d, %d), Y shape (%d, %d)",
			len(X), len(X[0]), len(Y), len(Y[0]))

		t.Logf("Sample X data:")
		for i := 0; i < min(5, len(X)); i++ {
			t.Logf("  [%.2f %.2f]", X[i][0], X[i][1])
		}

		tempMin, tempMax := findMinMax(X, 0)
		durMin, durMax := findMinMax(X, 1)
		t.Logf("Temperature Max, Min pre normalization: %.2f, %.2f", tempMax, tempMin)
		t.Logf("Duration Max, Min pre normalization: %.2f, %.2f", durMax, durMin)

		// Normalize the data
		normalizer := mathutil.NewNormalizer()
		XNorm, err := normalizer.FitTransform(X)
		if err != nil {
			t.Fatalf("Normalization failed: %v", err)
		}

		// Print range after normalization
		tempMinNorm, tempMaxNorm := findMinMax(XNorm, 0)
		durMinNorm, durMaxNorm := findMinMax(XNorm, 1)
		t.Logf("Temperature Max, Min post normalization: %.2f, %.2f", tempMaxNorm, tempMinNorm)
		t.Logf("Duration Max, Min post normalization: %.2f, %.2f", durMaxNorm, durMinNorm)

		// Create network
		config := network.NetworkConfig{
			LayerSizes:   []int{2, 3, 1},
			Activations:  []string{"sigmoid", "sigmoid"},
			LearningRate: 0.01,
		}

		mlp, err := network.NewMLP(config)
		if err != nil {
			t.Fatalf("Network creation failed: %v", err)
		}

		// Set pretrained weights and baises
		layer1, err := mlp.Layer(0)
		if err != nil {
			t.Fatalf("Failed to get layer 1: %v", err)
		}
		layer1.Weights[0] = []float64{-8.93, -0.1}
		layer1.Weights[1] = []float64{0.29, -7.32}
		layer1.Weights[2] = []float64{12.9, 10.81}
		layer1.Biases[0] = -9.82
		layer1.Biases[1] = -9.28
		layer1.Biases[2] = 0.96

		layer2, err := mlp.Layer(1)
		if err != nil {
			t.Fatalf("Failed to get layer 1: %v", err)
		}

		layer2.Weights[0] = []float64{-31.18, -27.59, -32.56}
		layer2.Biases[0] = 15.41

		testCases := []struct {
			name        string
			temperature float64
			duration    float64
			expected    float64
		}{
			{"positive_example", 200.0, 13.9, 0.7},
			{"negative_example", 200.0, 17.0, 0.3},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				testInput := [][]float64{{tc.temperature, tc.duration}}

				t.Logf("testInput pre normalization: %v", testInput)
				testInputNorm, err := normalizer.Transform(testInput)
				if err != nil {
					t.Fatalf("Failed to normalize test input: %v", err)
				}

				t.Logf("Noramlized Temperature %v, normalized duration %v", tc.temperature, tc.duration)
				t.Logf("testInputNorm %v", testInputNorm)
				t.Logf("testInputNorm[0] %v", testInputNorm[0])

				output, err := mlp.Forward(testInputNorm[0])
				if err != nil {
					t.Fatalf("Forward pass failed: %v", err)
				}
				t.Logf("testInputNorm :%v", testInputNorm[0])

				probability := output[0]
				decision := 0.0
				if probability > 0.5 {
					decision = 1.0
				}

				t.Logf("Input: [%.1f, %.1f] -> Normalized: [%.3f, %.3f] -> Probability: %.6f -> Decision: %.0f",
					tc.temperature, tc.duration,
					testInputNorm[0][0], testInputNorm[0][1],
					probability, decision)
			})
		}

		t.Run("batch_predictions", func(t *testing.T) {
			batchSize := min(10, len(X))
			correctPredictions := 0

			t.Logf("Batch predictions on first %d training samples:", batchSize)
			t.Logf("Sample | Input [Temp, Dur] | Probability | Prediction | Actual | Correct")
			t.Logf("-------|------------------|-------------|------------|--------|--------")

			for i := 0; i < batchSize; i++ {
				output, err := mlp.Forward(XNorm[i])
				if err != nil {
					t.Errorf("Batch prediction %d failed: %v", i, err)
					continue
				}

				probability := output[0]
				predicted := 0.0
				if probability >= 0.5 {
					predicted = 1.0
				}

				correct := ""
				if predicted == Y[i][0] {
					correctPredictions++
					correct = "✓"
				} else {
					correct = "✗"
				}

				t.Logf("  %2d   | [%5.1f, %4.1f] |   %.6f   |    %.0f      |   %.0f   |   %s",
					i+1, X[i][0], X[i][1], probability, predicted, Y[i][0], correct)
			}

			accuracy := float64(correctPredictions) / float64(batchSize)
			t.Logf("Batch accuracy: %.2f%% (%d/%d)", accuracy*100, correctPredictions, batchSize)
		})

		// Verify architecture
		t.Run("verify_architecture", func(t *testing.T) {
			if mlp.GetLayerCount() != 2 {
				t.Errorf("Expected 2 layers, got %d", mlp.GetLayerCount())
			}
			if mlp.InputSize() != 2 {
				t.Errorf("Expected input size 2, got %d", mlp.InputSize())
			}
			if mlp.OutputSize() != 1 {
				t.Errorf("Expected output size 1, got %d", mlp.OutputSize())
			}

			t.Logf("Network architecture verified: %s", mlp.String())
		})
	})
}

func TestNormalizationBehavior(t *testing.T) {
	t.Run("matches_expected_ranges", func(t *testing.T) {
		X, _ := coffeedata.LoadCoffeeData()

		normalizer := mathutil.NewNormalizer()
		XNormalized, err := normalizer.FitTransform(X)
		if err != nil {
			t.Fatalf("Normalization failed: %v", err)
		}

		// Check that normalized data has approximately zero mean and unit variance
		for feature := 0; feature < 2; feature++ {
			sum := 0.0
			sumSquares := 0.0

			for i := 0; i < len(XNormalized); i++ {
				val := XNormalized[i][feature]
				sum += val
				sumSquares += val * val
			}

			mean := sum / float64(len(XNormalized))
			variance := sumSquares/float64(len(XNormalized)) - mean*mean
			stddev := math.Sqrt(variance)

			featureName := "Temperature"
			if feature == 1 {
				featureName = "Duration"
			}
			t.Logf("%s: mean=%.6f, stddev=%.6f", featureName, mean, stddev)

			// Mean should be very close to 0
			if math.Abs(mean) > 1e-10 {
				t.Errorf("Feature %d mean = %v; should be ≈ 0", feature, mean)
			}

			// Standard deviation should be very close to 1
			if math.Abs(stddev-1.0) > 1e-10 {
				t.Errorf("Feature %d stddev = %v; should be ≈ 1", feature, stddev)
			}
		}

		// Log the actual ranges achieved
		tempMin, tempMax := findMinMax(XNormalized, 0)
		durMin, durMax := findMinMax(XNormalized, 1)
		t.Logf("Actual normalized ranges:")
		t.Logf("  Temperature: [%.2f, %.2f]", tempMin, tempMax)
		t.Logf("  Duration: [%.2f, %.2f]", durMin, durMax)
	})
}

func BenchmarkCoffeeRoasting(b *testing.B) {
	// Set up
	X, _ := coffeedata.LoadCoffeeData()
	normalizer := mathutil.NewNormalizer()
	XNorm, _ := normalizer.FitTransform(X)

	config := network.NetworkConfig{
		LayerSizes:   []int{2, 3, 1},
		Activations:  []string{"sigmoid", "sigmoid"},
		LearningRate: 0.01,
	}

	mlp, _ := network.NewMLP(config)

	// Set pretrained weights (abbreviated for benchmark)
	layer1, _ := mlp.Layer(0)
	layer1.Weights[0] = []float64{-8.93, -0.1}
	layer1.Weights[1] = []float64{0.29, -7.32}
	layer1.Weights[2] = []float64{12.9, 10.81}
	layer1.Biases = []float64{-9.82, -9.28, 0.96}

	layer2, _ := mlp.Layer(1)
	layer2.Weights[0] = []float64{-31.18, -27.59, -32.56}
	layer2.Biases[0] = 15.41

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for j := 0; j < len(XNorm); j++ {
			_, _ = mlp.Forward(XNorm[j])
		}
	}
}
