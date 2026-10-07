# 3-d Reconstruction from Two Images

Inspired by "Multiview Geometry in Computer Vision" by Hartley and Zisserman (HZ).

This repository captures an exploratory attempt to build a 3-d reconstruction pipeline. I began by taking two images of the same scene,
with a camera positioned at slightly different angles. I then hand-selected matching points (to the best of my ability) in each 
image -- see `matches.pdf` for the result.

`triangulate.py` is the core of the pipeline, and it implements the algorithms described in HZ for reconstruction with two views:
- As seen in `matches.pdf` we have a set of point correspondences $x_i \leftrightarrow x_i'$, which are the input into the pipeline. We need 8 or more correspondences for this pipeline.
- We then start by estimating the fundamental matrix, $F$, using the normalized 8-point algorithm (HZ Ch. 11).
- Next we calculate (estimate) the epipoles, $e$ and $e'$, associated with the two cameras.
- We then construct the "canonical camera matrices" $P$ and $P'$, using the epipoles (HZ Ch. 9).
- The triangulation pipeline then computes corrected point correspondences $\hat{x}_i \leftrightarrow \hat{x}_i'$ using the "optimal solution to triangulation" discussed in HZ Ch. 12.
- Finally, the corrected points $\hat{x}_i \leftrightarrow \hat{x}_i'$ and camera matrices $P, P'$ can be fed into the direct linear triangulation algorithm to produce a set of 3-d points.

Note that the canonical camera matrices lie in a 3-d spaced related to the true Euclidean space by an unknown projective transformation. Thus, the reconstructed scene can only be accurate up to this projective ambiguity. Further information about the cameras or scene would need to be incorporated to improve the reconstruction.
The 3-d reconstruction can be seen in `plot3d.ipynb`.

This project was inspired by a course project focused on algebraic vision. See the write-up here for more background and an
introduction to algebraic vision: <https://hans-elliott99.github.io/projects/math-380/>

-Hans
