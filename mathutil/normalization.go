package mathutil

import (
	"fmt"
	"math"
)

type Normalizer struct {
	mean   []float64
	stddev []float64
	fitted bool
}

func NewNormalizer() *Normalizer {
	return &Normalizer{
		fitted: false,
	}
}

// Learn the mean and standard deviation for each feature
func (n *Normalizer) Fit(data [][]float64) error {
	if len(data) == 0 || len(data[0]) == 0 {
		return fmt.Errorf("data cannot be empty")
	}

	numFeatures := len(data[0])
	numSamples := len(data)

	n.mean = make([]float64, numFeatures)
	n.stddev = make([]float64, numFeatures)

	// Calculate mean for each feature
	for i := range n.mean {
		sum := 0.0
		for j := range numSamples {
			sum += data[j][i]
		}
		n.mean[i] = sum / float64(numSamples)
	}

	// Calculate standard deviation for each feature
	for i := range n.stddev {
		sumSquares := 0.0
		for j := range numSamples {
			diff := data[j][i] - n.mean[i]
			sumSquares += diff * diff
		}
		n.stddev[i] = math.Sqrt(sumSquares / float64(numSamples))

		// Handle zero variance case (but not NaN/Inf cases)
		// Only set to 1.0 if it's exactly zero (constant finite values)
		if n.stddev[i] == 0 {
			n.stddev[i] = 1.0
		}
	}

	n.fitted = true
	return nil
}

// Transform scales the given data using the calculated mean and standard deviation
func (n *Normalizer) Transform(data [][]float64) ([][]float64, error) {
	if !n.fitted {
		return nil, fmt.Errorf("normalizer must be fitted before transform")
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	numFeatures := len(data[0])
	if numFeatures != len(n.mean) {
		return nil, fmt.Errorf("feature count mismatch: normalizer was fitted on %d features, data has %d", len(n.mean), numFeatures)
	}

	normalized := make([][]float64, len(data))
	for i, sample := range data {
		if len(sample) != numFeatures {
			return nil, fmt.Errorf("inconsistent feature count at sample %d: expected %d, got %d", i, numFeatures, len(sample))
		}
		normalized[i] = make([]float64, numFeatures)
		for j := range sample {
			normalized[i][j] = (sample[j] - n.mean[j]) / n.stddev[j]
		}
	}

	return normalized, nil
}

func (n *Normalizer) FitTransform(data [][]float64) ([][]float64, error) {
	if err := n.Fit(data); err != nil {
		return nil, fmt.Errorf("failed to fit data: %w", err)
	}
	return n.Transform(data)
}

func (n *Normalizer) GetParams() (mean []float64, stddev []float64, fitted bool) {
	if !n.fitted {
		return nil, nil, false
	}

	meanCopy := make([]float64, len(n.mean))
	stddevCopy := make([]float64, len(n.stddev))
	copy(meanCopy, n.mean)
	copy(stddevCopy, n.stddev)

	return meanCopy, stddevCopy, true
}
