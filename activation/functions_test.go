package activation

import (
	"math"
	"testing"
)

func TestReLUForward(t *testing.T) {
	tests := []struct {
		input, expected float64
	}{
		{-1.0, 0.0},
		{0.0, 0.0},
		{2.0, 2.0},
	}

	for _, test := range tests {
		actual := ReLU{}.Forward(test.input)
		if actual != test.expected {
			t.Errorf("ReLU.Forward(%v) = %v, expected %v", test.input, actual, test.expected)
		}
	}
}

func TestReLUName(t *testing.T) {
	expected := "ReLU"
	actual := ReLU{}.String()
	if expected != actual {
		t.Errorf("ReLU.String() = %s, expected %s", actual, expected)
	}
}

func TestSigmoidForward(t *testing.T) {
	tests := []struct {
		input, expected float64
	}{
		{0.0, 0.5},
		{1, 0.7310586},
		{-1, 0.2689414},
		{10, 0.9999546},
		{-10, 0.0000454},
	}

	epsilon := 0.0001

	for _, test := range tests {
		actual := Sigmoid{}.Forward(test.input)
		if math.Abs(test.expected-actual) > epsilon {
			t.Errorf("Sigmoid.Forward(%v) = %v, expected %v", test.input, actual, test.expected)
		}
	}
}

func TestSigmoidName(t *testing.T) {
	expected := "Sigmoid"
	actual := Sigmoid{}.String()
	if expected != actual {
		t.Errorf("Sigmoid.String() = %s, expected %s", actual, expected)
	}
}

func TestLinearForward(t *testing.T) {
	tests := []struct {
		input, expected float64
	}{
		{1.0, 1.0},
		{-1.0, -1.0},
		{20.0, 20.0},
		{-100.0, -100.0},
	}

	for _, test := range tests {
		actual := Linear{}.Forward(test.input)
		if actual != test.expected {
			t.Errorf("Linear.Forward(%v) = %v, expected %v", test.input, actual, test.expected)
		}
	}
}

func TestLinearName(t *testing.T) {
	expected := "Linear"
	actual := Linear{}.String()
	if expected != actual {
		t.Errorf("Linear.String() = %s, expected %s", actual, expected)
	}
}

func TestTanhForward(t *testing.T) {
	tests := []struct {
		input, expected float64
	}{
		{0.0, 0.0},
		{-1.0, -0.7616},
		{1.0, 0.7616},
		{10.0, 1.0},
		{-10.0, -1.0},
		{100.0, 1.0},
		{-100.0, -1.0},
	}

	epsilon := 0.0001

	for _, test := range tests {
		actual := Tanh{}.Forward(test.input)
		if math.Abs(actual-test.expected) > epsilon {
			t.Errorf("Tanh.Forward(%v) = %v, expected %v", test.input, actual, test.expected)
		}
	}
}

func TestTanhString(t *testing.T) {
	expected := "Tanh"
	actual := Tanh{}.String()
	if expected != actual {
		t.Errorf("Tanh.String() = %s, expected %s", actual, expected)
	}
}

func TestNewActivation(t *testing.T) {
	tests := []struct {
		name     string
		expected string
	}{
		{"relu", "ReLU"},
		{"sigmoid", "Sigmoid"},
		{"linear", "Linear"},
		{"tanh", "Tanh"},
		{"unknown", "Linear"}, // default case
	}

	for _, test := range tests {
		actual := NewActivation(test.name)
		if actual.String() != test.expected {
			t.Errorf("NewActivation(%q).String() = %q, want %q", test.name, actual.String(), test.expected)
		}
	}
}
