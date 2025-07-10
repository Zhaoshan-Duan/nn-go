package coffeedata

import (
	"math/rand"
	"time"
)

func LoadCoffeeData() ([][]float64, [][]float64) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	X := make([][]float64, 200)

	for i := range X {
		X[i] = make([]float64, 2)
	}

	for i := range X {
		tempRandom := r.Float64()
		durationRandom := r.Float64()

		X[i][1] = durationRandom*4 + 11.5
		X[i][0] = tempRandom*(295-150) + 150
	}

	Y := make([][]float64, 200)
	for i := range Y {
		Y[i] = make([]float64, 1)
	}

	for i := 0; i < len(X); i++ {
		t := X[i][0]
		d := X[i][1]

		y := -3/(260-175)*t + 21

		if t > 175 && t < 260 && d > 12 && d < 15 && d <= y {
			Y[i][0] = 1 // Good coffee
		} else {
			Y[i][0] = 0 // Bad coffee
		}
	}
	return X, Y
}
