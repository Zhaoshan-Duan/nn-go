## Implementation Progress Tracker

### Component Checklist

| Component                | Implemented | Unit Tested | Notes                |
|--------------------------|:-----------:|:-----------:|----------------------|
| Activation Functions     | [x]         | [x]         |ReLU, Sigmoid, Tanh, Linear; value tests|
| Layer Struct             | [x]         | [x]         |Init, shape, activation, value ranges|
| Layer Forward Pass       | [x]         | [x]         |Output shape, activations, input checks|
| Network Struct (MLP)     | [x]         | [x]         |Init, error handling, tests|
| Forward Propagation      | [ ]         | [ ]         |                      |
| Backward Propagation     | [ ]         | [ ]         |                      |
| Training Loop            | [ ]         | [ ]         |                      |
| Data Loading             | [ ]         | [ ]         |                      |
| Evaluation/Metrics       | [ ]         | [ ]         |                      |
| CLI/Plotting             | [ ]         | [ ]         |                      |

### Details & Notes

- **Activation Functions:**
  - [x] Implement ReLU, Sigmoid, Tanh, Linear
  - [x] United tested

- **Layer Struct:**
  - [x] Define weights, biases, activation
  - [x] Test initialization
  - [x] Test Forward Propagation

- Network (MLP) Struct:
  - [x] Define layers, learning rate
  - [x] Test initialization    
  - [ ] Test Forward Propagation

---

**Instructions:**
- Mark `[x]` when a component is implemented or tested.
- Add notes for bugs, refactors, or ideas.
