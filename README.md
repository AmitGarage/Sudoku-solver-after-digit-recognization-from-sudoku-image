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

***CNN Model used for training***

- nn.Conv2d(1,20,kernel_size=5)
  - max_pool2d(,2)
  - relu
- nn.Conv2d(20,40,kernel_size=5)
  - nn.Dropout2d()
  - max_pool2d(,2)
  - relu
- nn.Linear(640,120)
- dropout
- nn.Linear(120,10)
- log_softmax

Image after masking and finding grids coordinates
<p align="center">
  <img src="/Sudoku3.jpeg" width="350" title="Image after masking and finding grids coordinates">
</p>

Image after suduko grid was cropped out from orignal image
<p align="center">
  <img src="/Sudoku2.jpeg" width="350" title="Image after suduko grid was cropped out from orignal image">
</p>
