package mathutil

func DotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("DotProduct: length mismatch!!")
	}

	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}
