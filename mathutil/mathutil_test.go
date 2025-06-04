package mathutil

import "testing"

func TestDotProduct(t *testing.T) {
	tests := []struct {
		a, b     []float64
		expected float64
	}{
		{[]float64{1, 2, 3}, []float64{4, 5, 6}, 32}, // 1*4 + 2*5 + 3*6 = 32
		{[]float64{0, 0, 0}, []float64{1, 2, 3}, 0},
		{[]float64{1.5, -2.0}, []float64{2.0, 3.0}, 1.5*2.0 + (-2.0)*3.0}, // 3.0 - 6.0 = -3.0
		{[]float64{}, []float64{}, 0},
	}

	for _, testCase := range tests {
		actual := DotProduct(testCase.a, testCase.b)
		if actual != testCase.expected {
			t.Errorf("DotProduct(%v, %v) = %v; want %v", testCase.a, testCase.b, actual, testCase.expected)
		}
	}
}

func TestDotProductLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch, got none")
		}
	}()
	DotProduct([]float64{1, 2}, []float64{1})
}

func TestDotProductEdgeCases(t *testing.T) {
	// Empty slices
	if actual := DotProduct([]float64{}, []float64{}); actual != 0 {
		t.Errorf("DotProduct of empty slices = %v; want 0", actual)
	}

	// Single element
	if actual := DotProduct([]float64{2}, []float64{3}); actual != 6 {
		t.Errorf("DotProduct({2}, {3}) = %v; want 6", actual)
	}

	// Negative values
	if actual := DotProduct([]float64{-1, 2}, []float64{3, -4}); actual != (-1*3 + 2*-4) {
		t.Errorf("DotProduct({-1,2}, {3,-4}) = %v; want %v", actual, -1*3+2*-4)
	}
}
