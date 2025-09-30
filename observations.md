### `([85*85, 20, 25, 6], (30, 20, 2))`

```csv
6.60377358490566, 48.113207547169814, 45.28301886792453, 45.28301886792453, 45.28301886792453, 34.90566037735849, 50.943396226415096, 26.41509433962264, 52.83018867924528, 45.28301886792453
```

### `([85*85, 20, 25, 6], (30, 20, 0.2))`

```csv
6.60377358490566, 43.39622641509434, 42.45283018867924, 43.39622641509434, 45.28301886792453, 49.056603773584904, 49.056603773584904, 55.660377358490564, 57.54716981132076, 54.71698113207547
```

### Extreme plays with learning rate

If the learning rate is too high (>5) the accuracy will behave quite randomly and eventually reach values closer to zero (in a few tests it would stabilize on 6.60%).

## Some things to notice

- the accuracy going down doesn't imply that the cost is going up, since accuracy is discrete (either you guess the chess piece or you don't) and is thus only loosely related to the cost
- if the overall cost grows, then we are moving from one concavity to one other, potentially reaching a new local minimum
- the shallower the network (less layers), the hardest it is to minimize the cost function; however, the accuracy can sometimes even improve
