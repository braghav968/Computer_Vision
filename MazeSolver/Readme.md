<p>
The project deals with finding the shortest path in a given maze by mentioning start points and end points from the mouse dynamically on the maze.
The input image is given as "Input_Maze_Image.jpg".
We have used morphological operations such as erosion, dilation, skeletization for creating the thin image for maze solving which is given by "Thinned_Maze_Image.jpg".
After applying the A-star algorithm, the output of the image is given as "Output_Shortest_Path_Image.png".
The model gives accurate output for input maze.<br>
<H4> Here, A* Algorithm is used for finding the shortest path between Start position and End position </H4>
 <h4> Technologies used:
   <h5> 1. OpenCV-Python <br>
     2. Image morphological operators such as: Dilation, Erosion, Skeletonization<br>
     3. Algorithm: A* algorithm </h4></h4>
</p>
<h1> A* Algorithm </h1>
<p align="center">
<img src = "https://media.geeksforgeeks.org/wp-content/uploads/a_-search-algorithm-1.png" height = 500>
</p>
<h1> Input Image of the Maze </h1>
<p align="center">
<img src = "https://cdn.statically.io/gh/braghav968/Titanic-Kaggle/master/images/Input_Maze_Image.png" height = 500>
</p>
<h1> Processed Maze Image After Thinning </h1>
<p align="center">
<img src = "https://cdn.statically.io/gh/AtharvaKalsekar/Computer_Vision/c445e755/MazeSolver/maze_thinned.png" height = 500>
</p>
<h1> Output Image Showing Shortest Path From Start To End Point </h1>
<p align="center">
<img src = "https://cdn.statically.io/gh/AtharvaKalsekar/Computer_Vision/c445e755/MazeSolver/Output_Shortest_Path_Image.jpg" height = 500>
</p>
<br><h3> The green path shows the shortest route. <h4>
