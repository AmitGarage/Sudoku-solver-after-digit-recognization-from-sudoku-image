# Sudoku-solver-after-digit-recognization-from-sudoku-image
Sudoku will be extracted from image and digits to be recognized for further solving sudoku

- Trained a CNN model on MNIST dataset for digit recognizer and achived an accuracy of 97.87 % .
- Below steps taken for extracting sudoku from a given image :
  - Computed a threshold mask image based on local pixel neighborhood and extracted image above the threshold value to avoid any noise present in image.
  - Extracted contours from image and sorted them in descending order.
  - Traversed contour and approximated a polygonal with the specified tolerance to find exact coordinates of sudoku grids.
  - Coordinates received was transformed over 270 x 270 matrix.
  - Removed borders of 1's or 0's.
  - Cropped images into 30 x 30 shape.
- Cropped images were used detect digit after border removal.
- Boxes which were blank and needs to be predicated were given 0 as digit.
- Used backtracking to solve sudoku.
